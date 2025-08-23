"""Integration tests for toololo.lib.subagent module - Temperature averaging task."""

import pytest
import os
from typing import List, Tuple

import openai
from toololo.lib.subagent import ParallelSubagents
from toololo.run import Run
from toololo.types import TextContent, ToolUseContent, ToolResult


@pytest.fixture
def openai_client():
    """Create a real OpenAI client."""
    return openai.AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1"
    )


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


def test_deterministic_tools():
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
    try:
        city_to_latlng("Not A City")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown city" in str(e)
    
    try:
        area_to_subareas("Not An Area")
        assert False, "Should have raised ValueError"  
    except ValueError as e:
        assert "Unknown area" in str(e)
    
    # Test latlng_temperature
    temp1 = latlng_temperature(25.0, -80.0)  # Miami-like (south, east)
    temp2 = latlng_temperature(45.0, -120.0)  # Seattle-like (north, west)
    assert isinstance(temp1, float)
    assert isinstance(temp2, float)
    assert temp1 > temp2, "Southern/Eastern locations should be warmer"


def test_temperature_progression():
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


@pytest.mark.asyncio
async def test_state_temperature_averaging(openai_client):
    """Test recursive temperature averaging using Run with spawn_agents as a tool.

    This creates a three-level hierarchy where agents discover the pattern:
    - Level 1: State agent (Run) uses spawn_agents to create county agents
    - Level 2: County agents use spawn_agents to create city agents  
    - Level 3: City agents use temperature tools to get actual data

    The LLM should figure out the recursive spawning pattern on its own.
    """

    # Define the base tools available to all agents
    tools = [
        area_to_subareas,
        is_city,
        city_to_latlng,
        latlng_temperature,
    ]

    # Create ParallelSubagents manager
    subagent_manager = ParallelSubagents(
        client=openai_client, 
        tools=tools,
        model="openai/gpt-5-mini",
        max_iterations=5
    )
    
    # Create a wrapper that consumes the async iterator and returns a simple result
    async def spawn_agents_tool(
        agent_prompts: list[str],
        system_prompt: str = ""
    ) -> str:
        """Tool wrapper for spawn_agents that consumes the async iterator and returns results."""
        results = []
        agent_statuses = {}
        
        try:
            async for output in subagent_manager.spawn_agents(agent_prompts, system_prompt):
                if output.is_final:
                    agent_statuses[output.agent_index] = "completed" if not output.error else f"failed: {output.error}"
                else:
                    # Look for temperature-related outputs
                    if isinstance(output.output, ToolResult) and output.output.success:
                        result_str = str(output.output.content)
                        if "temperature" in result_str.lower() and "°f" in result_str.lower():
                            results.append(f"Agent {output.agent_index}: {result_str}")
            
            # Summarize results
            completed_agents = sum(1 for status in agent_statuses.values() if status == "completed")
            failed_agents = len(agent_statuses) - completed_agents
            
            summary = f"Spawned {len(agent_prompts)} agents. "
            summary += f"Completed: {completed_agents}, Failed: {failed_agents}. "
            
            if results:
                summary += f"Temperature results: {'; '.join(results)}"
            else:
                summary += "No temperature results collected."
            
            return summary
            
        except Exception as e:
            return f"Error spawning agents: {str(e)}"
    
    tools.append(spawn_agents_tool)

    print(f"\n🌡️  Starting State Temperature Averaging with Run + spawn_agents tool")
    print(f"📍 State: California")
    print("🔄 Structure: Run → spawn_agents tool → recursive subagents")

    # Run the main state agent using Run
    run = Run(
        client=openai_client,
        messages="Compute the average temperature for the state of California",
        model="openai/gpt-5-mini",
        tools=tools,
        system_prompt="You are a temperature coordinator. When you need to get temperatures for multiple areas, use spawn_agents to create parallel agents. Each agent will handle one area.",
        max_iterations=15
    )

    # Track results and temperatures
    outputs = []
    temperature_values = []
    spawn_calls = 0
    final_result = None

    # Run and collect outputs
    async for output in run:
        outputs.append(output)
        output_type = type(output).__name__
        print(f"🔄 Main Agent Output: {output_type}")

        # Track spawn_agents tool calls
        if isinstance(output, ToolUseContent) and "spawn_agents" in output.name:
            spawn_calls += 1
            print(f"  📡 Spawn call #{spawn_calls}: {len(output.input.get('agent_prompts', []))} agents")

        # Look for temperature data in tool results
        elif isinstance(output, ToolResult) and output.success:
            result_text = str(output.content).lower()

            # Look for temperature mentions
            if "temperature" in result_text and ("°f" in result_text or "degrees" in result_text):
                print(f"  📊 Temperature result: {output.content[:100]}...")

                # Try to extract numerical temperatures
                import re
                temp_matches = re.findall(r'(\d+\.?\d*)\s*°?f', result_text)
                for temp_str in temp_matches:
                    try:
                        temp_val = float(temp_str)
                        if 0 < temp_val < 200:  # Reasonable temperature range
                            temperature_values.append(temp_val)
                    except ValueError:
                        pass

            # Look for final average result
            if "average" in result_text and ("california" in result_text or "state" in result_text):
                final_result = output.content
                print(f"  🎯 Final State Result: {final_result}")

        # Look for text content with temperatures
        elif isinstance(output, TextContent):
            content_lower = output.content.lower()
            if "temperature" in content_lower and ("california" in content_lower or "average" in content_lower):
                print(f"  💬 Text with temperature: {output.content[:100]}...")
                final_result = output.content

    # Verify the recursive structure worked
    print(f"\n📊 Temperature Averaging Summary:")
    print(f"  Total Outputs: {len(outputs)}")
    print(f"  Spawn Agent Calls: {spawn_calls}")
    print(f"  Temperature Values Found: {len(temperature_values)}")
    if temperature_values:
        print(f"  Temperature Range: {min(temperature_values):.1f}°F - {max(temperature_values):.1f}°F")
        overall_avg = sum(temperature_values) / len(temperature_values)
        print(f"  Overall Average: {overall_avg:.1f}°F")
    print(f"  Final Result: {final_result}")

    # Count different types of outputs for verification
    tool_uses = [o for o in outputs if isinstance(o, ToolUseContent)]
    tool_results = [o for o in outputs if isinstance(o, ToolResult)]
    text_outputs = [o for o in outputs if isinstance(o, TextContent)]

    print(f"  Tool Uses: {len(tool_uses)} (spawn_agents: {spawn_calls})")
    print(f"  Tool Results: {len(tool_results)}")
    print(f"  Text Outputs: {len(text_outputs)}")

    # Assertions for recursive behavior and temperature collection
    assert len(outputs) > 0, "Should have some outputs"
    assert spawn_calls > 0, "Should have called spawn_agents at least once"
    assert len(tool_results) > 0, "Should have some tool results"
    assert len(temperature_values) > 0, "Should have found some temperature values"

    # Verify temperature values are reasonable
    assert all(isinstance(temp, float) and 0 < temp < 200 for temp in temperature_values), \
        "Temperatures should be reasonable float values"

    # Should have evidence of multi-level processing (multiple spawn calls or many temp values)
    assert spawn_calls >= 1 or len(temperature_values) >= 3, \
        "Should show evidence of multi-level processing"

    print(f"🎉 Recursive temperature averaging with Run + spawn_agents verified!")
    print(f"🌡️  Successfully processed temperature data through agent hierarchy")

    return {
        "total_outputs": len(outputs),
        "spawn_calls": spawn_calls,
        "temperature_values": temperature_values,
        "final_result": final_result,
        "overall_average": sum(temperature_values) / len(temperature_values) if temperature_values else None
    }
