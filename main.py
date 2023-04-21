def nested_dict():
    """ a function to store and perform nested dictionary operations in the form of tabular data"""

    #  JSON file is dictionary inside a list
    # houses_rowwise = [
    #     {
    #         "price_aprox_usd": 115910.26,
    #         "surface_covered_in_m2": 128,
    #         "rooms": 4,
    #     },
    #     {
    #         "price_aprox_usd": 48718.17,
    #         "surface_covered_in_m2": 210,
    #         "rooms": 3,
    #     },
    #     {
    #         "price_aprox_usd": 28977.56,
    #         "surface_covered_in_m2": 58,
    #         "rooms": 2,
    #     },
    #     {
    #         "price_aprox_usd": 36932.27,
    #         "surface_covered_in_m2": 79,
    #         "rooms": 3,
    #     },
    #     {
    #         "price_aprox_usd": 83903.51,
    #         "surface_covered_in_m2": 111,
    #         "rooms": 3,
    #     },
    # ]

    #  columnwise set of list inside dictionary
    houses_columnwise = {
        "price_aprox_usd": [115910.26, 48718.17, 28977.56, 36932.27, 83903.51],
        "surface_covered_in_m2": [128.0, 210.0, 58.0, 79.0, 111.0],
        "rooms": [4.0, 3.0, 2.0, 3.0, 3.0],
    }

    house_lst = []
    avg_price = 0
    #  loop to iterate and add the price per square meter in the house

    # for house in houses_rowwise:
    #
    #     #  house["price_m2"]=house[ "price_aprox_usd"] / house[ "surface_covered_in_m2"]
    #    house_lst.append(house["price_aprox_usd"])
    #
    # # return houses_rowwise
    # # return house_lst
    #
    # for price in house_lst:
    #     avg_price += price
    # avg_price = avg_price / len(house_lst)
    # return avg_price

    # mean_house_price = sum(houses_columnwise["price_aprox_usd"]) /len(houses_columnwise["price_aprox_usd"])
    # return mean_house_price

    price = houses_columnwise["price_aprox_usd"]  # defining a variable to extract list from the above dictionary
    area = houses_columnwise["surface_covered_in_m2"]
    price_per_m2 = []  # blank list to add the calculated price

    for p, a in zip(price,
                    area):  # loop to extract the values in list by using zip method to merge two lists in tuple format
        price_m2 = p / a
        price_per_m2.append(price_m2)
    houses_columnwise["price_per_m2"] = price_per_m2
    return houses_columnwise


#  -------------------------------------------------------------------------------------------------------------------------------------------
#  ----------------------------------------------------------------------------------------------------------------------------------------------------------

#  start of main method to run all the above listed functions

if __name__ == '__main__':
    print(nested_dict())
