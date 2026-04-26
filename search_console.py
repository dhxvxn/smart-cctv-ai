from intent_manager import IntentManager
from query_engine import QueryEngine
from video_player import play_event

engine = QueryEngine()
intent_manager = IntentManager()

print("AI CCTV Search Console")
print("Type 'exit' to quit")


def ask_session_mode_once():
    print("\nSelect search mode:")
    print("1. Single Camera Events")
    print("2. Multi Camera Events")

    while True:
        choice = input("Enter choice (1/2): ").strip().lower()
        if choice in {"1", "single", "single camera"}:
            return "single"
        if choice in {"2", "multi", "multi camera"}:
            return "multi"
        if choice == "exit":
            return None
        print("Invalid choice. Please enter 1, 2, or exit.")


session_mode = ask_session_mode_once()
if session_mode is None:
    raise SystemExit(0)

while True:

    query = input(f"\nAsk something ({session_mode} mode): ").strip()

    if query.lower() == "exit":
        raise SystemExit(0)

    intent_manager.set_intent(query)
    filters = intent_manager.get_filters()
    results = engine.run_query(filters=filters, session_mode=session_mode)

    if not results:
        print("No results.")
        continue

    print("\nResults:")

    for i, r in enumerate(results):
        cameras = r.get("cameras") or [r["camera_id"]]
        camera_label = ",".join(str(camera) for camera in cameras if camera is not None)
        print(
            f"{i}: {r['display_label']}={r['display_value']} | camera {camera_label or '-'} | "
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
