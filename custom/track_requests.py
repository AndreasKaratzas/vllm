"""
Global request tracker to compare R1, R2, R3 across cache operations.
"""

request_counter = 0
request_data = {}

def track_request(request_id, event_type, data=None):
    """Track events for each request."""
    global request_counter, request_data
    
    if request_id not in request_data:
        request_counter += 1
        request_data[request_id] = {
            'request_num': request_counter,
            'events': [],
        }
    
    request_data[request_id]['events'].append({
        'type': event_type,
        'data': data or {},
    })
    
    req_num = request_data[request_id]['request_num']
    print(f"[REQ_{req_num:02d}] {event_type}: {data}")

def get_request_num(request_id):
    """Get the request number (R1, R2, R3, etc.)"""
    return request_data.get(request_id, {}).get('request_num', 0)

def print_summary():
    """Print summary of all requests."""
    print("\n" + "="*100)
    print("REQUEST TRACKING SUMMARY")
    print("="*100)
    for req_id, data in sorted(request_data.items(), key=lambda x: x[1]['request_num']):
        print(f"\nRequest {data['request_num']} ({req_id[-12:]}):")
        for event in data['events']:
            print(f"  - {event['type']}: {event['data']}")
