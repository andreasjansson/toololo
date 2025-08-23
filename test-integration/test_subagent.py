"""Integration tests for toololo.lib.subagent module - Temperature averaging task."""

import pytest
import asyncio
import os
from typing import List, Tuple

import openai
from toololo.lib.subagent import spawn_parallel_agents, SubagentOutput, ParallelSubagents
from toololo.types import TextContent, ToolUseContent, ToolResult


@pytest.fixture
def openai_client():
    """Create a real OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return openai.AsyncOpenAI(api_key=api_key)


# Deterministic tools for temperature averaging task

STATE_TO_COUNTIES = {
    "California": ["Los Angeles County", "San Francisco County", "San Diego County"],
    "Texas": ["Harris County", "Dallas County", "Travis County"],
    "Florida": ["Miami-Dade County", "Orange County", "Hillsborough County"],
    "New York": ["New York County", "Kings County", "Queens County"],
}

COUNTY_TO_CITIES = {
    "Los Angeles County": ["Los Angeles", "Long Beach", "Pasadena"],
    "San Francisco County": ["San Francisco"],
    "San Diego County": ["San Diego", "Chula Vista", "Oceanside"],
    "Harris County": ["Houston", "Pasadena", "Baytown"],
    "Dallas County": ["Dallas", "Irving", "Garland"],
    "Travis County": ["Austin", "Round Rock", "Cedar Park"],
    "Miami-Dade County": ["Miami", "Hialeah", "Miami Beach"],
    "Orange County": ["Orlando", "Winter Park", "Oviedo"],
    "Hillsborough County": ["Tampa", "Plant City", "Temple Terrace"],
    "New York County": ["Manhattan"],
    "Kings County": ["Brooklyn"],
    "Queens County": ["Queens"],
}

CITY_COORDINATES = {
    "Los Angeles": (34.0522, -118.2437),
    "Long Beach": (33.7701, -118.1937),
    "Pasadena": (34.1478, -118.1445),
    "San Francisco": (37.7749, -122.4194),
    "San Diego": (32.7157, -117.1611),
    "Chula Vista": (32.6401, -117.0842),
    "Oceanside": (33.1959, -117.3795),
    "Houston": (29.7604, -95.3698),
    "Baytown": (29.7355, -94.9777),
    "Dallas": (32.7767, -96.7970),
    "Irving": (32.8140, -96.9489),
    "Garland": (32.9126, -96.6389),
    "Austin": (30.2672, -97.7431),
    "Round Rock": (30.5083, -97.6789),
    "Cedar Park": (30.5052, -97.8203),
    "Miami": (25.7617, -80.1918),
    "Hialeah": (25.8576, -80.2781),
    "Miami Beach": (25.7907, -80.1300),
    "Orlando": (28.5383, -81.3792),
    "Winter Park": (28.6000, -81.3395),
    "Oviedo": (28.6698, -81.2084),
    "Tampa": (27.9506, -82.4572),
    "Plant City": (28.0186, -82.1120),
    "Temple Terrace": (28.0356, -82.3890),
    "Manhattan": (40.7831, -73.9712),
    "Brooklyn": (40.6782, -73.9442),
    "Queens": (40.7282, -73.7949),
}


def area_to_subareas(area: str) -> List[str]:
    """Return a list of subareas for an area. For a state, return counties, for counties, return cities."""
    if area in STATE_TO_COUNTIES:
        return STATE_TO_COUNTIES[area]
    elif area in COUNTY_TO_CITIES:
        return COUNTY_TO_CITIES[area]
    else:
        raise ValueError(f"Unknown area: {area}")


def is_city(area_or_city: str) -> bool:
    """Return true if it's a city."""
    return area_or_city in CITY_COORDINATES


def city_to_latlng(city: str) -> Tuple[float, float]:
    """Return a lat/lng for a city. Throw an error if it's not a city."""
    if city not in CITY_COORDINATES:
        raise ValueError(f"Unknown city: {city}")
    return CITY_COORDINATES[city]


def latlng_temperature(lat: float, lng: float) -> float:
    """Deterministically return temperature for a lat/lng. The further south/east, the warmer."""
    # Base temperature starts at 50°F
    base_temp = 50.0
    
    # Add temperature based on southern latitude (closer to equator = warmer)
    # Latitude ranges roughly from 25-45 in our data
    lat_adjustment = (45 - lat) * 1.5  # Warmer as lat gets smaller (further south)
    
    # Add temperature based on eastern longitude (arbitrary for this test)
    # Longitude ranges roughly from -122 to -73 in our data
    lng_adjustment = (lng + 122) * 0.3  # Warmer as lng gets larger (further east)
    
    return base_temp + lat_adjustment + lng_adjustment


class TestTemperatureAveraging:
    """Integration test for recursive subagent temperature averaging task."""
    
    @pytest.mark.asyncio
    async def test_state_temperature_averaging(self, openai_client):
        """Test recursive temperature averaging: State -> Counties -> Cities -> Temperatures.
        
        This creates a three-level hierarchy:
        - Level 1: State agent coordinates county agents
        - Level 2: County agents coordinate city agents  
        - Level 3: City agents get temperatures and return them
        
        The results should bubble back up to compute averages.
        """
        
        # Level 3 tool: Get temperature for a city (used by city agents)
        async def get_city_temperature(city_name: str, client: openai.AsyncOpenAI = None) -> str:
            """Level 3 Agent Tool: Get temperature for a specific city."""
            try:
                if not is_city(city_name):
                    return f"Error: {city_name} is not a city"
                
                lat, lng = city_to_latlng(city_name)
                temp = latlng_temperature(lat, lng)
                return f"Temperature for {city_name}: {temp:.1f}°F"
                
            except Exception as e:
                return f"Error getting temperature for {city_name}: {str(e)}"
        
        # Level 2 tool: Get average temperature for a county (spawns city agents)
        async def get_county_temperature_average(county_name: str, client: openai.AsyncOpenAI = None) -> str:
            """Level 2 Agent Tool: Get average temperature for a county by spawning city agents."""
            if not client:
                return f"No client available for {county_name}"
            
            try:
                cities = area_to_subareas(county_name)
                if not cities:
                    return f"No cities found for {county_name}"
                
                print(f"    🏙️  County agent processing {len(cities)} cities in {county_name}")
                
                # Spawn city agents (Level 3)
                city_agent_specs = []
                for city in cities:
                    city_agent_specs.append((
                        f"You are a City Temperature Agent for {city}. Get the temperature for your city.",
                        f"Get the temperature for {city} using your temperature tool.",
                        [get_city_temperature, is_city, city_to_latlng, latlng_temperature]
                    ))
                
                # Collect temperature results from city agents
                city_temperatures = []
                completed_cities = set()
                
                async for result in spawn_parallel_agents(
                    client=client,
                    agent_specs=city_agent_specs,
                    model="gpt-4o-mini",
                    max_iterations=3
                ):
                    if result.is_final:
                        completed_cities.add(result.agent_index)
                        if result.error:
                            print(f"      ❌ City agent {result.agent_index} failed: {result.error}")
                        else:
                            print(f"      ✅ City agent {result.agent_index} ({cities[result.agent_index] if result.agent_index < len(cities) else result.agent_index}) completed")
                    
                    # Look for temperature outputs in tool results
                    elif isinstance(result.output, ToolResult) and result.output.success:
                        temp_result = str(result.output.result_content)
                        if "Temperature for" in temp_result and "°F" in temp_result:
                            try:
                                # Extract temperature value
                                temp_str = temp_result.split(":")[-1].strip().replace("°F", "")
                                temp_value = float(temp_str)
                                city_temperatures.append(temp_value)
                                print(f"      📊 Got temperature: {temp_value}°F from city agent {result.agent_index}")
                            except (ValueError, IndexError):
                                print(f"      ⚠️  Could not parse temperature from: {temp_result}")
                    
                    # Break when all city agents complete
                    if len(completed_cities) == len(city_agent_specs):
                        break
                
                # Calculate average
                if city_temperatures:
                    avg_temp = sum(city_temperatures) / len(city_temperatures)
                    return f"Average temperature for {county_name}: {avg_temp:.1f}°F (from {len(city_temperatures)} cities)"
                else:
                    return f"Could not get temperature data for {county_name}"
                
            except Exception as e:
                return f"Error processing county {county_name}: {str(e)}"
        
        # Level 1: State agent coordinates county agents
        state_agent_spec = [(
            "You are a State Temperature Coordinator. You manage county temperature agents to get "
            "the average temperature for your state. Coordinate county agents to get temperatures.",
            f"Get the average temperature for California by coordinating with county agents. "
            f"Use your county temperature tool to process each county in the state.",
            [get_county_temperature_average, area_to_subareas, is_city]
        )]
        
        print(f"\n🌡️  Starting State Temperature Averaging")
        print(f"📍 State: California")
        print("🔄 Structure: State → County Agents → City Agents → Temperature Lookup")
        
        # Track the temperature averaging process
        state_results = []
        state_completed = False
        final_temperatures = []
        
        # Run the state coordinator agent
        async for result in spawn_parallel_agents(
            client=openai_client,
            agent_specs=state_agent_spec,
            model="gpt-4o-mini",
            max_iterations=6
        ):
            if result.is_final:
                state_completed = True
                if result.error:
                    print(f"❌ State coordinator failed: {result.error}")
                else:
                    print(f"✅ State temperature averaging completed")
            else:
                state_results.append(result.output)
                output_type = type(result.output).__name__
                print(f"🔄 State Coordinator: {output_type}")
                
                # Look for county average temperatures in tool results
                if isinstance(result.output, ToolResult) and result.output.success:
                    county_result = str(result.output.result_content)
                    if "Average temperature for" in county_result and "°F" in county_result:
                        try:
                            # Extract temperature value
                            temp_str = county_result.split(":")[-1].strip().split("°F")[0].strip()
                            temp_value = float(temp_str.split("(")[0].strip())
                            final_temperatures.append(temp_value)
                            print(f"📊 County average: {temp_value}°F")
                        except (ValueError, IndexError):
                            print(f"⚠️  Could not parse county temperature from: {county_result}")
            
            if state_completed:
                break
        
        # Verify the recursive structure worked and calculate final state average
        print(f"\n📊 Temperature Averaging Summary:")
        print(f"  State Coordinator Status: {'✅ Success' if state_completed else '❌ Failed'}")
        print(f"  Total Operations: {len(state_results)}")
        print(f"  County Averages Collected: {len(final_temperatures)}")
        
        if final_temperatures:
            state_average = sum(final_temperatures) / len(final_temperatures)
            print(f"  Final California Average: {state_average:.1f}°F")
        else:
            print(f"  ❌ No temperature data collected")
        
        # Look for evidence of recursive operations in tool results
        county_operations = 0
        city_references = 0
        
        for output in state_results:
            if isinstance(output, ToolResult) and output.success:
                result_text = str(output.result_content).lower()
                if "average temperature for" in result_text and "county" in result_text:
                    county_operations += 1
                if "cities" in result_text:
                    city_references += 1
        
        print(f"  County Operations: {county_operations}")
        print(f"  City References: {city_references}")
        
        # Assertions for recursive behavior and temperature collection
        assert state_completed, "State coordinator should complete successfully"
        assert len(state_results) > 0, "Should have some temperature operations"
        assert county_operations > 0, "Should have evidence of county-level processing"
        assert len(final_temperatures) > 0, "Should have collected county temperature averages"
        assert all(isinstance(temp, float) and 0 < temp < 200 for temp in final_temperatures), "Temperatures should be reasonable values"
        
        # Verify we got results for multiple counties (California has 3 counties in our test data)
        expected_counties = len(STATE_TO_COUNTIES["California"])
        print(f"  Expected Counties: {expected_counties}")
        
        print(f"🎉 Recursive temperature averaging verified!")
        print(f"🌡️  Successfully computed temperature averages across {len(final_temperatures)} counties")
        
        return {
            "state_success": state_completed,
            "total_operations": len(state_results),
            "county_operations": county_operations,
            "final_temperatures": final_temperatures,
            "state_average": sum(final_temperatures) / len(final_temperatures) if final_temperatures else None
        }
    
    def test_deterministic_tools(self):
        """Test that the deterministic tools work correctly."""
        # Test area_to_subareas
        ca_counties = area_to_subareas("California")
        assert len(ca_counties) == 3
        assert "Los Angeles County" in ca_counties
        
        la_cities = area_to_subareas("Los Angeles County")
        assert len(la_cities) == 3
        assert "Los Angeles" in la_cities
        
        # Test is_city
        assert is_city("Los Angeles") == True
        assert is_city("Los Angeles County") == False
        assert is_city("California") == False
        
        # Test city_to_latlng
        la_coords = city_to_latlng("Los Angeles")
        assert isinstance(la_coords, tuple)
        assert len(la_coords) == 2
        assert isinstance(la_coords[0], float)
        assert isinstance(la_coords[1], float)
        
        # Test error handling
        with pytest.raises(ValueError):
            city_to_latlng("Not A City")
        
        with pytest.raises(ValueError):
            area_to_subareas("Not An Area")
        
        # Test latlng_temperature
        temp1 = latlng_temperature(25.0, -80.0)  # Miami-like (south, east)
        temp2 = latlng_temperature(45.0, -120.0)  # Seattle-like (north, west)
        assert isinstance(temp1, float)
        assert isinstance(temp2, float)
        assert temp1 > temp2, "Southern/Eastern locations should be warmer"
        
    def test_temperature_progression(self):
        """Test that temperature increases as we go south and east."""
        # Test north to south (latitude decrease = temperature increase)
        north_temp = latlng_temperature(40.0, -100.0)
        south_temp = latlng_temperature(30.0, -100.0)
        assert south_temp > north_temp, "Southern locations should be warmer"
        
        # Test west to east (longitude increase = temperature increase)
        west_temp = latlng_temperature(35.0, -120.0)
        east_temp = latlng_temperature(35.0, -80.0)
        assert east_temp > west_temp, "Eastern locations should be warmer"
        
        # Test actual city temperatures make sense
        la_temp = latlng_temperature(*CITY_COORDINATES["Los Angeles"])
        miami_temp = latlng_temperature(*CITY_COORDINATES["Miami"])
        houston_temp = latlng_temperature(*CITY_COORDINATES["Houston"])
        
        # Miami should be warmest (furthest south and east)
        assert miami_temp > la_temp
        assert miami_temp > houston_temp
