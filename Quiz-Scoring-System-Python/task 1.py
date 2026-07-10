questions_pool = [
    {
        "question": "What is the primary library used for data manipulation in Python?",
        "options": "A) NumPy\nB) Pandas\nC) Matplotlib\nD) Scikit-learn",
        "correct_answer": "B",
        "marks": 5,
        "penalty": 2
    },
    {
        "question": "Which programming language is predominantly used in Data Science alongside Python?",
        "options": "A) Java\nB) C++\nC) R\nD) HTML",
        "correct_answer": "C",
        "marks": 5,
        "penalty": 2
    },
    {
        "question": "What does SQL stand for?",
        "options": "A) Structured Query Language\nB) Simple Query Language\nC) Sequential Query Language\nD) Standard Query Logic",
        "correct_answer": "A",
        "marks": 5,
        "penalty": 1
    }
]

total_score = 0
current_question_number = 1
total_questions = len(questions_pool)

print("--- Welcome to the Data Science Intro Quiz ---")
print(f"Total Questions: {total_questions}\n")

for q in questions_pool:
    print(f"Question {current_question_number}: {q['question']}")
    print(q['options'])
    
    user_answer = input("Your answer (A, B, C, or D): ").strip().upper()
    
    if user_answer == q['correct_answer']:
        print(f"Correct. You awarded +{q['marks']} marks.\n")
        total_score += q['marks']
    else:
        print(f"Wrong answer. The correct answer was {q['correct_answer']}.")
        print(f"Penalty applied: -{q['penalty']} marks.\n")
        total_score -= q['penalty']
        
    current_question_number += 1

print("--- Quiz Results ---")
print(f"Your Final Score: {total_score}")

if total_score >= 10:
    print("Feedback: Excellent job! You have a solid foundational grasp.")
elif total_score > 0:
    print("Feedback: Good effort! A little more review and you'll be sharp.")
else:
    print("Feedback: Keep practicing. Data science requires continuous learning!")