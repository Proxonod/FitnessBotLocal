import asyncio
from src.orchestrator import Orchestrator

async def test():
    orc = Orchestrator()
    # Simuliert User-Input "300g Gouda"
    result = await orc.handle_text(
        telegram_id=123,
        user_name="Test",
        text="300g Gouda"
    )
    print(result)

asyncio.run(test())