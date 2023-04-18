def nested_dict ():

    """ a function to store and perform nested dictionary operatioons"""
    #  JSON file is dictionary inside a list
    houses_rowwise = [
        {
            "price_aprox_usd": 115910.26,
            "surface_covered_in_m2": 128,
            "rooms": 4,
        },
        {
            "price_aprox_usd": 48718.17,
            "surface_covered_in_m2": 210,
            "rooms": 3,
        },
        {
            "price_aprox_usd": 28977.56,
            "surface_covered_in_m2": 58,
            "rooms": 2,
        },
        {
            "price_aprox_usd": 36932.27,
            "surface_covered_in_m2": 79,
            "rooms": 3,
        },
        {
            "price_aprox_usd": 83903.51,
            "surface_covered_in_m2": 111,
            "rooms": 3,
        },
    ]

 #  loop to iterate and add the price per square meter in the house
    for house in houses_rowwise:
        house["price_m2"]=house[ "price_aprox_usd"] / house[ "surface_covered_in_m2"]

    return houses_rowwise




#  -------------------------------------------------------------------------------------------------------------------------------------------
#  ----------------------------------------------------------------------------------------------------------------------------------------------------------

#  start of main method to run all the above listed functions

if __name__ == '__main__':
    print(nested_dict ())


