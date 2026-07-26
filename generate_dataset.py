"""
generate_dataset.py
--------------------
Builds a labeled spam/ham (not-spam) message dataset for training.

In a real interview project you'd typically use the public "SMS Spam
Collection" dataset (UCI / Kaggle). Since this environment has no internet
access, this script generates a realistic synthetic dataset using common
spam patterns (prize/lottery, urgency, credit offers, phishing links) and
common everyday (ham) messages, with randomized variation so the model
has to learn genuine patterns rather than memorize exact strings.

Run:
    python generate_dataset.py
Produces:
    data/spam_dataset.csv  (columns: label, message)
"""

import csv
import random

random.seed(42)

spam_templates = [
    "Congratulations! You've won a {prize} worth ${amount}. Claim now at {link}",
    "URGENT: Your account will be suspended. Verify immediately at {link}",
    "You have been selected for a FREE {prize}! Click {link} to claim your reward",
    "WINNER!! As a valued customer you have been selected to receive a ${amount} gift card. Call now",
    "Get rich quick! Earn ${amount} per week working from home. No experience needed. Reply YES",
    "Your loan of ${amount} has been APPROVED. No credit check required. Apply now at {link}",
    "FREE entry into our ${amount} weekly prize draw! Text WIN to enter now",
    "Limited time offer! Get {discount}% off on all products. Shop now at {link}",
    "You have 1 new voicemail from your bank regarding suspicious activity. Call {phone} immediately",
    "Congrats! You are eligible for a {prize}. To claim, send your bank details to {link}",
    "Hot singles in your area want to chat with you! Click {link} now",
    "Your package could not be delivered. Pay a redelivery fee of ${amount} at {link}",
    "CASH PRIZE ALERT: You've won ${amount} in our lucky draw. Claim within 24 hours at {link}",
    "Earn extra ${amount}/day with this simple trick! Doctors hate it. Click {link}",
    "Final notice: Your subscription payment of ${amount} failed. Update card details at {link}",
    "Act now! Only {discount}% seats left for our FREE webinar that will make you rich",
    "Dear customer, your OTP has expired, click {link} to verify your identity and avoid account lock",
    "You've been chosen for a mystery {prize} box! Just pay ${amount} shipping. Click {link}",
    "IRS Notice: You owe ${amount} in unpaid taxes. Pay immediately at {link} to avoid arrest",
    "Claim your inheritance of ${amount} from a distant relative. Contact {link} with your details",
]

ham_templates = [
    "Hey, are we still meeting for lunch at {time} tomorrow?",
    "Can you send me the report before {time}? Thanks!",
    "Don't forget to pick up milk and eggs on your way home",
    "The meeting has been rescheduled to {time} on Friday",
    "Happy birthday! Hope you have a wonderful day",
    "I'll be a few minutes late, stuck in traffic",
    "Can we reschedule our call to {time}?",
    "Thanks for helping me move last weekend, really appreciate it",
    "Just finished the project, sending it over for review now",
    "What time does the movie start tonight?",
    "Reminder: your dentist appointment is at {time} tomorrow",
    "Let's catch up over coffee this weekend if you're free",
    "The Wi-Fi at the office is down again, IT is looking into it",
    "Great job on the presentation today, the client loved it",
    "Can you review my code before I push it to the main branch?",
    "I'm heading to the gym, want to join?",
    "Please find attached the invoice for last month's services",
    "Mom said dinner is ready, come downstairs",
    "Our flight got delayed by two hours, landing around {time}",
    "Congratulations on your promotion! Well deserved",
    "Let me know if you need help studying for the exam",
    "I left my charger at your place, can I grab it tomorrow?",
    "The train was on time today, made it to work early",
    "Can you proofread this email before I send it to the manager?",
    "Just checking in — how did your interview go?",
]

prizes = ["iPhone 16", "vacation package", "gift card", "laptop", "smartwatch", "cash prize", "luxury cruise"]
amounts = ["500", "1000", "2500", "10000", "250", "5000", "750"]
discounts = ["50", "70", "80", "90", "60"]
links = ["bit.ly/claim-now", "tinyurl.com/win-big", "secure-verify.net", "reward-center.info", "claim-prize.co"]
phones = ["1-800-555-0199", "1-888-222-4567", "1-877-333-9911"]
times = ["3 PM", "10 AM", "6:30 PM", "noon", "9 AM", "5 PM", "8 PM"]


def fill(template):
    return template.format(
        prize=random.choice(prizes),
        amount=random.choice(amounts),
        link=random.choice(links),
        discount=random.choice(discounts),
        phone=random.choice(phones),
        time=random.choice(times),
    )


rows = []
for _ in range(8):
    for t in spam_templates:
        rows.append(("spam", fill(t)))
for _ in range(6):
    for t in ham_templates:
        rows.append(("ham", fill(t)))

random.shuffle(rows)

with open("data/spam_dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "message"])
    writer.writerows(rows)

print(f"Generated {len(rows)} messages -> data/spam_dataset.csv")
print(f"  spam: {sum(1 for r in rows if r[0]=='spam')}")
print(f"  ham : {sum(1 for r in rows if r[0]=='ham')}")
