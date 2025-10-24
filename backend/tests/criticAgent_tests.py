import sys
import os

# Ensure the project root 'backend' is on the path
# This assumes the test is run from the project root (D:\programming\python\Math_prof)
# using: python -m backend.tests.critic_agent_tests
from backend.app.agents.CriticAgent import CriticAgent
from backend.app.core.logger import setup_logging, logger

# --- Test Cases ---

test_cases = [
    {
        "description": "Correct and Clear Solution",
        "problem": "Solve for x: 2x + 5 = 17",
        "solution": """
To solve for x, we first isolate the term with x.
1. Subtract 5 from both sides: 2x = 17 - 5
2. Simplify: 2x = 12
3. Divide by 2: x = 12 / 2
4. Final Answer: x = 6
""",
        "expected_decision": "Accept"
    },
    {
        "description": "Mathematically Incorrect Solution",
        "problem": "Solve for x: 2x + 5 = 17",
        "solution": """
To solve for x:
1. Subtract 5 from both sides: 2x = 17 - 5
2. Simplify: 2x = 12
3. Divide by 2: x = 12 / 2
4. Final Answer: x = 5
""",
        "expected_decision": "Refine"
    },
    {
        "description": "Correct but Unclear (Missing Steps)",
        "problem": "Find the roots of the quadratic equation x^2 - 5x + 6 = 0.",
        "solution": "The roots of the equation are x = 2 and x = 3.",
        "expected_decision": "Refine"
    },
    {
        "description": "Incomplete Solution (Missing Median)",
        "problem": "Find the mean and median of the set: {10, 20, 30, 40, 50}",
        "solution": """
To find the mean, we sum the numbers and divide by the count.
Sum = 10 + 20 + 30 + 40 + 50 = 150
Count = 5
Mean = 150 / 5 = 30.
The mean is 30.
""",
        "expected_decision": "Refine"
    },
    {
        "description": "Correct but Overly Complex Solution",
        "problem": "What is 5 + 5?",
        "solution": """
Let x = 5. We are asked to find x + x, which is 2x.
We can define a function f(y) = 5 + y. We want to find f(5).
The derivative is f'(y) = 1.
Using Taylor expansion around y=0: f(y) = f(0) + f'(0)*y + ...
f(0) = 5. So f(y) = 5 + y.
Plugging in y=5, we get f(5) = 5 + 5 = 10.
The answer is 10.
""",
        "expected_decision": "Refine"
    },
    {
        "description": "Correct Geometry Solution",
        "problem": "Find the area of a circle with a radius of 5 units.",
        "solution": """
The formula for the area of a circle is A = pi * r^2.
Given:
- Radius (r) = 5 units
Steps:
1. Substitute the radius into the formula: A = pi * (5)^2
2. Calculate the square: A = pi * 25
3. Final Answer: The area is 25 * pi square units.
The solution is correct, clear, and complete.
""",
        "expected_decision": "Accept"
    },
    {
        "description": "Correct Full Statistics Solution",
        "problem": "Find the mean, median, and mode of the data set: {1, 2, 2, 3, 7}",
        "solution": """
1.  **Mean:** Sum of values / number of values.
    Sum = 1 + 2 + 2 + 3 + 7 = 15
    Count = 5
    Mean = 15 / 5 = 3.
2.  **Median:** The middle value of the ordered set.
    The set is already ordered: {1, 2, 2, 3, 7}.
    The middle value is the 3rd value, which is 2.
    Median = 2.
3.  **Mode:** The value that appears most frequently.
    The value 2 appears twice, all other values appear once.
    Mode = 2.
Final Answer: The mean is 3, the median is 2, and the mode is 2.
""",
        "expected_decision": "Accept"
    },
    {
        "description": "Correct Algebra (Distribution)",
        "problem": "Solve for y: 3(y - 1) = 9",
        "solution": """
To solve for y, we can first distribute the 3 on the left side.
1.  Distribute 3: 3*y - 3*1 = 9
    3y - 3 = 9
2.  Add 3 to both sides to isolate the 'y' term:
    3y = 9 + 3
    3y = 12
3.  Divide by 3 to solve for y:
    y = 12 / 3
    y = 4
Final Answer: y = 4.
""",
        "expected_decision": "Accept"
    }
]

# --- Test Runner (Simple Loop Style) ---

if __name__ == "__main__":
    # Set up logging
    try:
        setup_logging()
    except Exception as e:
        import logging
        logging.basicConfig(level=logging.INFO)
        logger.warning(f"Could not load custom logging config. Using basic config. Error: {e}")

    # Initialize the agent
    logger.info("--- Running Critic Agent Tests (Simple Style) ---")
    try:
        agent = CriticAgent()
        logger.info("Critic Agent initialized for testing.")
    except Exception as e:
        logger.error(f"Failed to initialize CriticAgent. Ensure model and API keys are set. Error: {e}")
        sys.exit(1)

    passed = 0
    failed = 0
    
    for test in test_cases:
        logger.info(f"\n[Testing]: {test['description']}")
        problem = test["problem"]
        solution = test["solution"]
        expected = test["expected_decision"]

        response = agent.critique(problem, solution)
        
        # Handle potential capitalization mismatch (e.g., 'decision' vs 'Decision')
        decision = response.get('decision') or response.get('Decision')
        feedback = response.get('feedback') or response.get('Feedback')

        if decision == expected:
            logger.info(f"  [PASS] Expected: '{expected}', Got: '{decision}'")
            passed += 1
        else:
            logger.error(f"  [FAIL] Expected: '{expected}', Got: '{decision}'")
            logger.error(f"  Feedback: {feedback}")
            failed += 1
            
    print("\n--- Test Summary ---")
    print(f"Total: {len(test_cases)}, Passed: {passed}, Failed: {failed}")

