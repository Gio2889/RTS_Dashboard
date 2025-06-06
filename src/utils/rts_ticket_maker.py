import random
from datetime import datetime, timedelta
import json
import requests
from time import sleep

# Configuration
NUM_TICKETS = 100
OUTPUT_FILE = 'data_center_tickets.json'
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"  # Change if using different model
NETWORKS = ['Core', 'Edge', 'Access', 'Backbone']
TICKET_PREFIX = "GBF"

# Local issue templates for fallback
ISSUE_TEMPLATES = [
    {
        "issue": "High CPU utilization causing service degradation",
        "fix": "1. Identify top processes:\n```bash\ntop -c\n```\n2. Kill runaway process:\n```bash\nkill -9 <PID>\n```\n3. Implement CPU limits:\n```bash\nsystemctl set-property <service> CPUQuota=75%\n```",
        "root_cause": "Misconfigured cron job spawning infinite processes"
    },
    {
        "issue": "Disk space exhaustion on /var partition",
        "fix": "1. Check disk usage:\n```bash\ndf -h\n```\n2. Clear old logs:\n```bash\nsudo find /var/log -type f -mtime +30 -delete\n```\n3. Resize partition:\n```bash\nsudo lvextend -r -L +10G /dev/mapper/vg-var\n```",
        "root_cause": "Unrotated application logs filling filesystem"
    },
    {
        "issue": "Network connectivity loss between racks",
        "fix": "1. Verify physical connections\n2. Check switch config:\n```bash\nssh switch01 show interfaces\n```\n3. Reset network interface:\n```bash\nsudo ifdown eth1 && sudo ifup eth1\n```",
        "root_cause": "Faulty network cable in TOR switch"
    },
    {
        "issue": "DNS resolution failures",
        "fix": "1. Check resolv.conf:\n```bash\ncat /etc/resolv.conf\n```\n2. Test DNS server:\n```bash\ndig @8.8.8.8 example.com\n```\n3. Restart DNS service:\n```bash\nsudo systemctl restart systemd-resolved\n```",
        "root_cause": "DNS server IP misconfigured during maintenance"
    }
]

def generate_issue_description():
    """Generate issue description using Ollama API or fallback to templates"""
    prompt = (
        "Generate a realistic Linux server issue in a data center environment with 3 sections: "
        "### Issue (problem description), "
        "### Fix Steps (numbered steps with bash commands in code blocks), "
        "### Root Cause (brief explanation). "
        "Use technical but concise language (60-100 words total)."
    )
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        print(f"Ollama error ({e}), using local template")
        template = random.choice(ISSUE_TEMPLATES)
        return (
            f"### Issue\n{template['issue']}\n\n"
            f"### Fix Steps\n{template['fix']}\n\n"
            f"### Root Cause\n{template['root_cause']}"
        )

def generate_timestamps():
    """Generate realistic timeline for a ticket"""
    start = datetime.now() - timedelta(days=random.randint(0, 30))
    
    # Response within 15-120 minutes
    response_delta = timedelta(minutes=random.randint(15, 120))
    response_time = start + response_delta
    
    # On-site decision (50% of tickets)
    if random.random() > 0.5:
        decision_delta = response_delta + timedelta(minutes=random.randint(5, 60))
        onsite_delta = decision_delta + timedelta(minutes=random.randint(15, 90))
        on_site_decision = start + decision_delta
        time_on_site = start + onsite_delta
    else:
        on_site_decision = None
        time_on_site = None
    
    # Resolution within 1-8 hours for closed tickets
    if random.random() > 0.3:  # 70% of tickets are closed
        resolution_delta = timedelta(hours=random.randint(1, 8))
        end_time = start + resolution_delta
    else:
        end_time = None
    
    return {
        "start_time": start,
        "response_time": response_time,
        "on_site_decision": on_site_decision,
        "time_on_site": time_on_site,
        "end_time": end_time
    }

def generate_tickets(num):
    """Generate ticket data with chronological numbering"""
    tickets = []
    print(f"Generating {num} tickets...")
    
    # Generate unsorted tickets
    for i in range(num):
        if i % 10 == 0:
            print(f"Generated {i}/{num} tickets")
        
        timestamps = generate_timestamps()
        responsibility = random.choices(
            ['yes', 'no', 'partial'],
            weights=[0.6, 0.2, 0.2]
        )[0]
        
        tickets.append({
            "start_time": timestamps["start_time"],
            "response_time": timestamps["response_time"],
            "on_site_decision": timestamps["on_site_decision"],
            "time_on_site": timestamps["time_on_site"],
            "issue_description": generate_issue_description(),
            "responsibility": responsibility,
            "network": random.choice(NETWORKS),
            "end_time": timestamps["end_time"],
            "main_office_ticket": random.choice([True, False])
        })
        sleep(0.1)  # Be gentle with Ollama server
    
    # Sort tickets chronologically
    tickets.sort(key=lambda x: x["start_time"])
    
    # Assign ticket numbers and process special cases
    ten_days_ago = datetime.now() - timedelta(days=10)
    
    for i, ticket in enumerate(tickets):
        # Assign chronological ticket number
        ticket["ticket_number"] = f"{TICKET_PREFIX}-{1000 + i}"
        
        # Last 15 tickets should have a high chance of being open
        if i >= len(tickets) - 15 and random.random() < 0.7:  # 70% chance
            ticket["end_time"] = None
        
        # Process main office ticket status
        if ticket["end_time"] is not None:  # Closed ticket
            if ticket["start_time"] < ten_days_ago:
                ticket["main_office_ticket"] = True
    
    print("Ticket generation complete!")
    return tickets

def convert_datetime_fields(ticket):
    """Convert datetime fields to ISO format strings for JSON serialization"""
    converted = ticket.copy()
    for field in ['start_time', 'response_time', 'on_site_decision', 'time_on_site', 'end_time']:
        if converted[field] is not None:
            converted[field] = converted[field].isoformat()
    return converted

def save_to_json(tickets, filename):
    """Save tickets to JSON file"""
    # Convert datetime objects to strings
    serializable_tickets = [convert_datetime_fields(t) for t in tickets]
    
    with open(filename, 'w') as f:
        json.dump(serializable_tickets, f, indent=2)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    tickets = generate_tickets(NUM_TICKETS)
    save_to_json(tickets, OUTPUT_FILE)