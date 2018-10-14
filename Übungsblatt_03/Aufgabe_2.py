print("Konvertierung in:")
conv = input(" Fahrenheit (1) oder Kelvin (2)")

if conv == 1:
    print("Konvertierung in Fahrenheit")
    celsius = float(input("Grad in Celsius"))
    if(celsius>=-273.15):
        fahrenheit = (celsius* 9/5) + 32
        print(str(celsius)+" Celsius sind "+str(fahrenheit)+" Fahrenheit")
    else:
        print ("Temp zu niedrig!")
elif conv == 2:
    print("Konvertierung in Kelvin")
    celsius = float(input("Grad in Celsius"))
    if (celsius >= -273.15):
        kelvin = celsius+273.15
        print(str(celsius) + " Celsius sind " + str(kelvin) + " Kelvin")
    else:
        print ("Temp zu niedrig!")
else:
    print("Falsche Eingabe!")