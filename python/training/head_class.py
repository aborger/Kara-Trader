import python.training.train_rnn as tr
import keras
from python.user_data.user import User

class Training_Model:
	TRAINSET = 0
	TESTSET = 0
	Users = User
	
	# init parameters are all possible variables
	# what_stocks, epochs, numbars, trainbarlength
	def __init__(self, ID, what_stocks, epochs, numbars, trainBarLength):
		self.model = tr.build_network(numbars)
		self.ID = ID
		self.performance = 0
		self.what_stocks = what_stocks
		self.epochs = epochs
		self.numbars = numbars
		self.trainBarLength = trainBarLength
		
		# For later use
		self.error = 0
		
		self._train()
		self._validate()
	
	# This function controls all models training process
	def oversee(TRAINSET, TESTSET, MODELS, new_model_name):
		Training_Model.TRAINSET = TRAINSET
		Training_Model.TESTSET = TESTSET
		
		
		Training_Model.Users.update_users() # Need to make more efficient in User
		
		# Start error sheet
		error_sheet = open(r"data/results/error.csv",'w')
		error_sheet.write('ID, Error,\n')
		error_sheet.close()
		
		# Write stock to sheet
		error_sheet = open(r"data/results/performance.csv",'w')
		error_sheet.write('ID, Error,\n')
		error_sheet.close()
	
	
		lowest_error = -1
		best_model = 0
		# Testing best epochs
		this_id = 0
		for epoch in range(0, 10):
			print('======================================')
			print('=                 ' + str(epoch) + ' / ' + str(10) + '             =')
			print('======================================')
			for num_stocks in range (1, 5):
				print('======================================')
				print('=                 ' + str(num_stocks) + ' / ' + str(5) + '             =')
				print('======================================')
				for what_stocks in range(0, 5):
					print('======================================')
					print('=                 ' + str(what_stocks) + ' / ' + str(5) + '             =')
					print('======================================')
					try:
						new_model = Training_Model(ID=this_id, what_stocks=[what_stocks,what_stocks + num_stocks,490,504], epochs=epoch, numbars=10, trainBarLength=1000)
					except IndexError:
						pass
					except:
						raise
					else:
						# Write stock to sheet
						error_sheet = open(r"data/results/error.csv",'a')
						error_sheet.write(str(new_model.ID) + ',' + str(new_model.error) + ',\n')
						error_sheet.close()
						#print('Error: ' + str(new_model.error))
							
						# Test if better than best
						if lowest_error == -1:
							lowest_error = new_model.error
							best_model = new_model
						elif new_model.error < lowest_error:
							lowest_error = new_model.error
							best_model = new_model
							
							# Write stock to sheet
							error_sheet = open(r"data/results/performance.csv",'a')
							error_sheet.write(str(new_model.ID) + ',' + str(new_model.error) + ',\n')
							error_sheet.close()
							print('New lowest error: ' + str(new_model.error))
							
						this_id += 1
			
				
				
		print('Lowest error of: ' + str(lowest_error))
		print('Saving model...')
		best_model.model.save(MODELS + new_model_name + '.h5')
		
		print('Viewing model...')
		best_model._test()
		
	
	def _train_collect(self, stock_num, type):
		from python.training.collect_data import collect
		# update users first to gain access to the api
		collect(Training_Model.Users.get_api(), stock_num, type, self.trainBarLength, Time='1D')
		
	def _train(self):
		for stock in range(self.what_stocks[0], self.what_stocks[1]):
			self._train_collect(stock, 'train')
			x_train, y_train = tr.prepare(Training_Model.TRAINSET, self.numbars, self.trainBarLength)
			self.model = tr.train_network(x_train, y_train, self.epochs, self.model)
			#print('Saving model...')
			#model.save('data/Trade-Model.h5')
				
	# returns average error over multiple stocks
	def _validate(self):
		error_sum = 0
		print('Getting Error...')
		for stock in range(self.what_stocks[2], self.what_stocks[3]):
			self._train_collect(stock, 'test')
			
			error = tr.validate_results(Training_Model.TRAINSET, Training_Model.TESTSET, self.model, self.numbars)
			#print('Error: ' + str(error))
			error_sum += error
			
		self.error = error_sum / (self.what_stocks[3] - self.what_stocks[2])
	# Shows image of prediction
	def _test(self):
		for stock in range(self.what_stocks[2], self.what_stocks[3]):
			self._train_collect(stock, 'test')

			tr.test_results(Training_Model.TRAINSET, Training_Model.TESTSET, self.model, self.numbars)
		
		