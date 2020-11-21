import stripe

TEST_KEY = "sk_test_51HLFZMEZVv1JoaylrbkPZTBcUTiq9QMbxeyRTYd4rncGS5NZFCEdhtEJftz8LpM7Mj7g8NXKEMQXEurCd2R0RR5y00r4KGsXAM"
LIVE_KEY = ""


stripe.api_key = TEST_KEY

customer_data = stripe.Customer.list()["data"]
customers = []
for customer in customer_data:
	customers.append(customer)
	print("Email: " + customer["email"])
	print("ID: " + customer["id"])
	
'''
charge = stripe.Charge.create(
	amount = 200,
	currency = 'usd',
	customer = customers[0]["id"],
	description = "A test charge of $2",
	receipt_email = customers[0]["email"],
	source = customers[0]["default_source"]
	)
print(charge)
'''
#print(stripe.Product.list())

'''
PRICE = 1200
subscript = stripe.Subscription.create(
	customer = customers[1]["id"],
	items=[{"price_data": {
		"currency": 'usd',
		"product": stripe.Product.list()["data"][0],
		"recurring": {'interval': 'month'},
		"unit_amount": PRICE
		}}]
)
print(subscript)
'''

#print(stripe.Subscription.list()["data"][0])
#print(stripe.Product.list()["data"][0])


'''
NEW_PRICE = 100
new_subscript = stripe.Subscription.modify(
	stripe.Subscription.list()["data"][0]["id"],
	items = [{
		"price_data": {
			"currency": 'usd',
			"product": stripe.Product.list()["data"][0]["id"],
			"recurring": {'interval': 'month'},
			"unit_amount": NEW_PRICE
			}
	}],
	proration_behavior = 'none'
)
print(new_subscript)
'''


print(stripe.Subscription.list(price='price_1HpgryEZVv1Joayldr6JkuLN'))