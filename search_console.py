from intent_manager import IntentManager
from query_engine import QueryEngine
from video_player import play_event

engine = QueryEngine()
intent_manager = IntentManager()

print("AI CCTV Search Console")
print("Type 'exit' to quit")

while True:

    query = input("\nAsk something: ")

    if query.lower() == "exit":
        break

    intent_manager.set_intent(query)
    filters = intent_manager.get_filters()
    results = engine.run_query(filters=filters)

    if not results:
        print("No results.")
        continue

    print("\nResults:")

    for i, r in enumerate(results):
        print(
            f"{i}: {r['display_label']}={r['display_value']} | camera {r['camera_id']} | "
            f"zone {r['zone_id']} | {r['object_type']} | "
            f"global {r.get('global_id', '-')} | track {r['track_id']}"
        )

    choice = input("\nSelect result number to play video (or press Enter): ")

    if choice.strip() == "":
        continue

    # 🔥 SAFE INDEX HANDLING
    try:
        idx = int(choice)

        if idx < 0 or idx >= len(results):
            print("Invalid selection.")
            continue

    except ValueError:
        print("Please enter a valid number.")
        continue

    result = results[idx]

    play_event(result)
