class Vehicle:
    def __init__(self, max_speed, mileage):
        self.max_speed = max_speed
        self.mileage = mileage

    def display_info(self):
         return f"Max_speed: {Toyota.max_speed}, Mileage: {Toyota.mileage}"   
        
Toyota = Vehicle(440, 7)
print(f"Max_speed: {Toyota.max_speed}, Mileage: {Toyota.mileage}" )




class Bus(Vehicle):
    def __init__(self, name, max_speed, mileage):
        super().__init__(max_speed, mileage)
        self.name = name

    def display_info(self):
        return f"Vehicle Names: {self.name}, Max_Speed: {self.max_speed}, mileage: {self.mileage} "


Vehicle_1 = Bus("Honda", "450", "9")
Vehicle_2 = Bus("Lamborghini", "890", "18" )   

print(Vehicle_1.display_info())
print(Vehicle_2.display_info())



class Lorry(Bus):
    def __init__(self, name, max_speed, mileage, capacity):
        super().__init__(name, max_speed, mileage)
        self.capacity = capacity 

    def seating_capacity(self, capacity):
        return f"The seating capacity of a {self.name} is {capacity} passenger"  


Vehicle_1 = Lorry("Honda", "450", "9", "50")  

print(Vehicle_1.seating_capacity())
