from typing import Dict
import asyncio

tasks: Dict[str, dict] = {}
hitl_events: Dict[str, asyncio.Event] = {}