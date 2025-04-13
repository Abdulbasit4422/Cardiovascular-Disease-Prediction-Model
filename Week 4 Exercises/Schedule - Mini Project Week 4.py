class schedule:
    def __init__(self, schedule = []):
        self.schedule = schedule 

    def add_schedule(self, new_schedule):
        self.schedule.append(new_schedule)
        return f"You've added {new_schedule} to your schedule"
    
    def get_schedule(self):
        return self.schedule
doc1 = schedule(["Monday", "Thursday"]) 
print(doc1.add_schedule("Saturday"))
print(doc1.get_schedule())       

