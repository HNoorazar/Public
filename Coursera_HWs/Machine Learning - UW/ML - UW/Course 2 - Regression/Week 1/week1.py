import graphlab
import numpy as np

# Load house sales data
sales = graphlab.SFrame('kc_house_data.gl/')


# Split data into training and testing
train_data,test_data = sales.random_split(.8,seed=0)

def simple_linear_regression(input_feature, output):
    no_examples = output.size()
    in_times_out = input_feature * output
    in_times_out_sum = in_times_out.sum()
    in_sum = input_feature.sum()
    out_sum = output.sum()
    slope_numerator = in_times_out_sum - ((in_sum * out_sum) / no_examples)
    
    in_squared = input_feature * input_feature
    in_squared_sum = in_squared.sum()
    slope_denominator = in_squared_sum - ( (in_sum * in_sum)/ no_examples)

    slope = slope_numerator / slope_denominator
    intercept = (out_sum / no_examples) - slope * (in_sum / no_examples)
    
    return (intercept, slope)
    
test_feature = graphlab.SArray(range(5))
test_output = graphlab.SArray(1 + 1*test_feature)
(test_intercept, test_slope) =  simple_linear_regression(test_feature, test_output)
    
sqft_intercept, sqft_slope = simple_linear_regression(train_data['sqft_living'], train_data['price'])
print "Intercept: " + str(sqft_intercept)
print "Slope: " + str(sqft_slope)

def get_regression_predictions(input_feature, intercept, slope):
    # calculate the predicted values:
    predicted_values = input_feature * slope + intercept
    
    return predicted_values

my_house_sqft = 2650
estimated_price = get_regression_predictions(my_house_sqft, sqft_intercept, sqft_slope)
print "The estimated price for a house with %d squarefeet is $%.2f" % (my_house_sqft, estimated_price)

def get_residual_sum_of_squares(input_featurer, outputr, interceptr, sloper):
    # First get the predictions
    predictions = get_regression_predictions(input_featurer, interceptr, sloper)

    # then compute the residuals (since we are squaring it doesn't matter which order you subtract)
    residuals = outputr - predictions

    # square the residuals and add them up
    residuals_square = residuals * residuals
    RSS = residuals_square.sum()
    return(RSS)

rss_prices_on_sqft = get_residual_sum_of_squares(train_data['sqft_living'], train_data['price'], sqft_intercept, sqft_slope)
print 'The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_sqft)

def inverse_regression_predictions(output, intercept, slope):
    # solve output = intercept + slope*input_feature for input_feature. 
    # Use this equation to compute the inverse predictions:
    estimated_feature = (output - intercept) / slope
    return estimated_feature
    
my_house_price = 800000
estimated_squarefeet = inverse_regression_predictions(my_house_price, sqft_intercept, sqft_slope)
print "The estimated squarefeet for a house worth $%.2f is %d" % (my_house_price, estimated_squarefeet)    


# Estimate the slope and intercept for predicting 'price' based on 'bedrooms'
bed_intercept, bed_slope = simple_linear_regression(train_data['bedrooms'], train_data['price'])

# Compute RSS when using bedrooms on TEST data:
rss_prices_on_sqft_bed = get_residual_sum_of_squares(test_data['bedrooms'], 
                                                     test_data['price'], bed_intercept, bed_slope)

print 'The RSS of predicting Prices based on bedroom count is (Test Data) : ' + str(rss_prices_on_sqft_bed)


# Compute RSS when using squarefeet on TEST data:
rss_prices_on_sqft = get_residual_sum_of_squares(test_data['sqft_living'], test_data['price'], sqft_intercept, sqft_slope)
print 'The RSS of predicting Prices based on Square Feet is (Test Data) : ' + str(rss_prices_on_sqft)

