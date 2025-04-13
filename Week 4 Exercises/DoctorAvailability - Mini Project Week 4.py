class DoctorAvailability:
    def __init__(self, dates):
        self.availability = {}
        self.dates = dates 
        for day in self.dates:
            self.availability[day] = 'Yes'
    def change_availability(self, day):
        self.availability[day] = 'No'
    def check_availability(self, day):
        if day in self.availability.keys():
            return self.availability.get(day)
        

Dr_Abdulbasit = DoctorAvailability(["Monday", "Wednesday", "Friday", "Saturday"])
print(Dr_Abdulbasit.check_availability("Friday"))

Dr_Abdulbasit.change_availability("Friday")
print(Dr_Abdulbasit.check_availability("Friday"))
                    
    