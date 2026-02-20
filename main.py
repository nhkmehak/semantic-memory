
from memory_engine import SemanticMemoryEngine

engine = SemanticMemoryEngine(similarity_threshold=0.80)

while True:
    user_input = input("\nEnter text (or type 'exit'): ")

    if user_input.lower() == "exit":
        break

    result = engine.store_or_match(user_input)

    print("Result:", result)

    print("Top related memories:")
    print(engine.retrieve(user_input))