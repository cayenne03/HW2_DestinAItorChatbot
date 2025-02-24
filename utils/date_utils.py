import re
from datetime import datetime, timedelta
from dateutil import parser
from dateutil.relativedelta import relativedelta

def parse_date_to_iso(date_text: str) -> str:
    """Convert various date formats to YYYY-MM-DD"""
    try:
        date_text = date_text.strip().lower()
        today = datetime.now()

        # handle "next day" -> tomorrow
        if date_text == "next day":
            return (today + timedelta(days=1)).strftime('%Y-%m-%d')

        # handle "next week" -> same day next week
        if date_text == "next week":
            return (today + timedelta(weeks=1)).strftime('%Y-%m-%d')

        # handle "thursday next week" -> specific weekday next week
        next_week_day = re.match(r'(\w+)\s+next\s+week', date_text)
        if next_week_day:
            day = next_week_day.group(1)
            days_map = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 
                       'friday': 4, 'saturday': 5, 'sunday': 6}
            if day in days_map:
                target_weekday = days_map[day]
                current_weekday = today.weekday()
                # First get to next week
                next_week = today + timedelta(weeks=1)
                # Then adjust to the target day
                days_to_add = (target_weekday - next_week.weekday()) % 7
                target_date = next_week + timedelta(days=days_to_add)
                return target_date.strftime('%Y-%m-%d')

        # handle "next month" -> same day next month
        if date_text == "next month":
            next_month = today + relativedelta(months=1)
            return next_month.strftime('%Y-%m-%d')

        # handle "in X months"
        in_months = re.match(r'in\s+(?:a|(\d+))\s+months?', date_text)
        if in_months:
            months = 1 if in_months.group(1) is None else int(in_months.group(1))
            target_date = today + relativedelta(months=months)
            return target_date.strftime('%Y-%m-%d')

        # handle "<N> days" and variations
        days_pattern = re.match(r'(\d+)\s*days?(?:\s*from\s*(now|today|tomorrow)?)?', date_text)
        if days_pattern:
            days = int(days_pattern.group(1))
            reference = days_pattern.group(2)
            base_date = today + timedelta(days=1) if reference == 'tomorrow' else today
            return (base_date + timedelta(days=days)).strftime('%Y-%m-%d')

        # handle "next <N> days"
        next_days_pattern = re.match(r'next\s+(\d+)\s*days?', date_text)
        if next_days_pattern:
            days = int(next_days_pattern.group(1))
            return (today + timedelta(days=days)).strftime('%Y-%m-%d')

        # handle "tomorrow"
        if date_text == 'tomorrow':
            return (today + timedelta(days=1)).strftime('%Y-%m-%d')

        # handle "next <day_of_week>"
        next_day_match = re.match(r'next\s+(\w+)', date_text)
        if next_day_match:
            day = next_day_match.group(1)
            days_map = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 
                        'friday': 4, 'saturday': 5, 'sunday': 6}
            if day in days_map:
                target_weekday = days_map[day]
                current_weekday = today.weekday()
                
                # Calculate days until next occurrence
                days_ahead = target_weekday - current_weekday
                if days_ahead <= 0:  # If target day has passed this week
                    days_ahead += 7  # Move to next week
                
                target_date = today + timedelta(days=days_ahead + 7)  # Add extra week for "next"
                return target_date.strftime('%Y-%m-%d')

        # handle "in/on the <number>(st/nd/rd/th) of next month"
        next_month_match = re.match(r'(?:in|on)?\s*(?:the)?\s*(\d{1,2})(?:st|nd|rd|th)?\s*(?:of)?\s*next\s*month', date_text)
        if next_month_match:
            day = int(next_month_match.group(1))
            target_month = today + relativedelta(months=1)
            while True:
                try:
                    return target_month.replace(day=day).strftime('%Y-%m-%d')
                except ValueError:
                    target_month += relativedelta(months=1)

        # handle standalone ordinals like "the 31st", "2nd"
        ordinal_match = re.match(r'(?:the)?\s*(\d{1,2})(?:st|nd|rd|th)?$', date_text)
        if ordinal_match:
            day = int(ordinal_match.group(1))
            target_date = today
            for _ in range(12):
                try:
                    candidate_date = target_date.replace(day=day)
                    if candidate_date >= today:
                        return candidate_date.strftime('%Y-%m-%d')
                    target_date += relativedelta(months=1)
                except ValueError:
                    target_date += relativedelta(months=1)

        # fallback: dateutil parser
        parsed_date = parser.parse(date_text, fuzzy=True)

        # if parsed date is today, return today
        if parsed_date.date() == today.date():
            return today.strftime('%Y-%m-%d')

        # if date is in the past and no year specified, shift to next year
        if parsed_date < today and not re.search(r'\b\d{4}\b', date_text):
            parsed_date += relativedelta(years=1)

        return parsed_date.strftime('%Y-%m-%d')

    except (ValueError, TypeError):
        return None



# # 🔥 TEST CASES
# test_inputs = [
#     "3 days",
#     "10 days from now",
#     "9 days from today",
#     "5 days from tomorrow",
#     "on Saturday",
#     "this Saturday",
#     "next 5 days",
#     "tomorrow",
#     "next day",
#     "next monday",
#     "next sunday",
#     "next week",
#     "thursday next week",
#     "2025-02-20",
#     "20 Feb 2025",
#     "February 20, 2025",
#     "02/20/2025",
#     "March 15",
#     "random text",
#     "27th",
#     "",
#     "in the 4th of next month",
#     "15 next month",
#     "next month",
#     "the 31st",
#     "1st of March",
#     None
# ]

# print("🚀 Running test cases...\n")
# for date_text in test_inputs:
#     result = parse_date_to_iso(date_text) if date_text else None
#     print(f"Input: {repr(date_text):<30} | Output: {result}")
