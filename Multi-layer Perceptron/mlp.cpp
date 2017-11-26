#include "mlp.h"    // header file of mlp.cpp
#include <time.h>	// time
#include <stdlib.h> // srand, time
#include <math.h>	// sqrt, exp
#include <stdio.h>
#include <string.h> //memset

/* Activation Function definition : Sigmoid, ReLU, LeakyReLU */

#define SIGMOID(x) (1./(1+exp(-x)))
//#define ReLU(x) ((x)>(0)?(x):(0))
//#define LeakyReLU(x) ((x)>(0.1*x)?(x):(0.1*x))

////////////////////////////////// Definition of a Node //////////////////////////////////////////////

node::node():w(0), b(0), w_b(0), activated(0){}  // constructor
node::~node() { delete w; }						 // destructor

void node::init(int prev_dim) // Initialise weight of a node
{
	// srand(time(NULL));
	b = 1;						  // Set bias as 1
	w = new float[prev_dim];      // weight vector from previous layer nodes to current node
	
	// Initialise w_b (which is bias vector, w0)
	w_b= 2 * (float(rand()) / (float)(RAND_MAX)) - 1; // random initialisation [-1, 1]
	//w_b = (float(rand()) / float(RAND_MAX)) / 2.f;

	// initialise weight vector (which is [w1, w2, ....wn] 
	for (int i = 0; i < prev_dim; i++)
	{
		// w[i] = (float(rand()) / float(RAND_MAX)) / 2.f;
		 w[i] = 2 * (float(rand()) / (float)(RAND_MAX)) - 1;	// random initalisation [-1, 1]
		// w[i] = rand() % (int)sqrt(12 / prev_dim) - sqrt(3 / prev_dim); // He's initalisation
	}
}

////////////////////////////////// Definition of a Layer //////////////////////////////////////////////

layer::layer() :cur_dim(0), neuron(0), prev_dim(0), prev_layer(0){}  // constructor
/*
layer::~layer() {													  // destructor
	delete neuron;
	delete prev_layer;
}
*/
/*	INIT FUNCTION

	[Parameter explanation]
	before_dim:= the number of nodes in the previous layer
	now_dim:= the number of nodes in the current layer

	[Layer Information]
	num_node: the number of nodes in the current layer (= now_dim)
	prev_dim: the number of nodes in the previous layer (= before_dim)

*/
void layer::init(int before_dim, int now_dim)
{
	neuron = new node *[now_dim];
	prev_layer = new float[prev_dim];

	cur_dim = now_dim;
	prev_dim = before_dim;

	/* Initialise Each Neuron */
	for (int i = 0; i < now_dim; i++)
	{
		neuron[i] = new node;
		neuron[i]->init(before_dim);
	}

}

/* ACTIVATION FUNCTION

	[Activation Function 3 Types defined below] 
	(1) Sigmoid
	(2) ReLU
	(3) LeakyReLU

*/
void layer::activation()
{
	for (int i = 0; i < cur_dim; i++)
	{
		float temp_before_activated = 0;
		for (int j = 0; j < prev_dim; j++)
		{
			temp_before_activated += neuron[i]->w[j] * prev_layer[j]; // Prev Layer(jth node) -> Current Layer(ith node)
		}
		temp_before_activated += neuron[i]->b*neuron[i]->w_b;		  // Add bias(b)*  bias_weight(w_b)
		
		/* Activation Function : SIGMOID Function */
		neuron[i]->activated = SIGMOID(temp_before_activated);
		//neuron[i]->activated = 1.f / (1.f + exp(temp_before_activated));

		/* Activation Function : ReLU Function */
		//neuron[i]->activated = ReLU(temp_before_activated);

		/* Activation Function : LeakyReLU Function */
		//neuron[i]->activated = LeakyReLU(temp_before_activated);
	}
}
////////////////////////////////// Definition of MLP //////////////////////////////////////////////

mlp::mlp() :hidden_layer(0), num_hidden_layer(0) {}   // constructor
//mlp::~mlp(){ delete hidden_layer; }
/*
mlp::~mlp() {										  // destructor
	for (int i = 0; i < num_hidden_layer; i++)
	{
		delete hidden_layer[i];
	}
	delete hidden_layer;
}
*/
/*	INIT FUNCTION

	[Parameter explanation]
	input_dim: the number of nodes in the input layer
	output_dim: the number of nodes in the output layer
	*hidden_dim: the array of number of nodes in each hidden layer (i.e., How many nodes are there in each hidden layer?)
 	hidden_cnt: the number of hidden layer (i.e., How many hidden layers are there? )
*/
void mlp::init(int input_dim, int input_neuron, int hidden_dim[], int hidden_cnt, int output_dim)
{
	/* 1. Initialise the  input layer */
	input_layer.init(input_dim, input_neuron); 
	
	/* 2. Initialise the hidden layer and the output layer 
	   (1) At least 1 hidden layer
	   (2) No hidden layer                                       
	*/

	// 2-(1) At least 1 hidden layer
	if (hidden_cnt >= 1)
	{
		hidden_layer = new layer *[hidden_cnt];
		num_hidden_layer = hidden_cnt;

		// Initialise all the hidden layers
		for (int k = 0; k < hidden_cnt; k++)
		{
			hidden_layer[k] = new layer;
			
			// The FIRST HIDDEN LAYER
			if (k == 0)
			{
				hidden_layer[k]->init(input_neuron, hidden_dim[k]);
			}

			// THE SECOND, THIRD, ... LAST hidden LAYER
			else
			{
				hidden_layer[k]->init(hidden_dim[k - 1], hidden_dim[k]);
			}
		}

		// Initialise the output layer
		output_layer.init(hidden_dim[hidden_cnt - 1], output_dim);
	}

	// 2-(2) No hidden layer (i.e., input and output layer ONLY )
	else
	{
		output_layer.init(input_neuron, output_dim);
	}
}


/*	FEED-FORWARD ALGORITHM

	Feed_forward and update the activated value of each neuron

	(1) Input feed_forward
	(2) Hidden feed_forward
	(3) Output feed_foward
	
	[Parameter explanation]
	input[]: input [x1, x2, ..., xn]

*/
void mlp::feed_forward(float input[])
{
	memcpy(input_layer.prev_layer, input, input_layer.prev_dim * sizeof(float));

	// 1-(1) Input feed_foward
	input_layer.activation();
	connection_update(-1);	 // update for the next one

	 // 1-(2) Hidden feed_forward
	if (num_hidden_layer >= 1)
	{
		for (int i = 0; i<num_hidden_layer; i++)
		{
			hidden_layer[i]->activation();
			connection_update(i); // update for the next one
		}
	}
	// 1-(3) Output feed_foward
	output_layer.activation();
}

/*	UPDATE FUNCTION
	Update the previous layer's activated nodes value

	[parameter explanation]
	what_layer: what hidden layer?

	(example)
	0:		the first hidden layer
	1:		the second hidden layer
	k-1:	the last hidden layer
	-1:		the input layer

*/
void mlp::connection_update(int what_layer)
{
	/*
	After activating the certain layer nodes value,
	we need to update the previous node value of the next layer to be connected with current and previous layer
	*/

	if (what_layer == -1) // after activating the input layer (feed_foward)
	{
		for (int i = 0; i < input_layer.cur_dim; i++)
		{
			if (hidden_layer) // At least one hidden layer
			{
				hidden_layer[0]->prev_layer[i] = input_layer.neuron[i]->activated;
			}
			else // NO hidden layer (i.e., only input and output layer exists)
			{
				output_layer.prev_layer[i] = input_layer.neuron[i]->activated;
			}
		}
	}

	else // after activating the hidden layer (feed_forward)
	{
		for (int i = 0; i < hidden_layer[what_layer]->cur_dim; i++)
		{
			if (what_layer < num_hidden_layer - 1) // not the last hidden layer
			{
				hidden_layer[what_layer + 1]->prev_layer[i] = hidden_layer[what_layer]->neuron[i]->activated;
			}
			else // last hidden layer
			{
				output_layer.prev_layer[i] = hidden_layer[what_layer]->neuron[i]->activated;
			}
		}
	}
}

/*	TRAIN FUNCTION

	1. Feed_forward and update the activated value of each neuron (defined above)

	2. Back-propagation ¡Ú¡Ú¡Ú (designed for sigmoid function)
		(1) Ouput layer to the last hidden layer (or input layer if no hidden layer exists)
		(2) Hidden layer[i] to Hidden layer[i-1]
		(3) Hidden layer[0] to Input layer

   [parameter explanation]
   *input: input [x1, x2, ...., xn]
   *real_value: output [y1, y2, ..., yn]
   learning_rate: how far the step goes

   [derivative of sigmoid function(x)]
   Let g(z):= 1/(1+exp(-z))
   Let f(z):= 1/g(z) then f(z):= 1+exp(-z)
   then f'(z):= -exp(-z)=1-f(z)
   also f'z()= -g'(z)/(g(z)^2)
   ¡æ g'(z)= -f'(z)*g(z)^2
   ¡æ g'(z)= (f(z)-1)*g(z)^2
   ¡æ g'(z)= (1/g(z)-1)*g(z)^2
   ¡æ g'(z)= g(z)-g(z)^2
   ¡æ g'(z)= g(z)[1-g(z)]	¡á 
*/

float mlp::train(float input[], float real_value[], float learning_rate)
{
	float loss_function = 0;  // loss function (= cost function, error function)
	float local_error;	      // local error
	float sum_transmited = 0; // summation of each error which will be transmitted to the previous layer
	float sum_of_layer = 0;	  // summation of new layer (firstly, reset to zero)
	float output;             // predicted output after activation

										/* 1. FEED-FORWARD ALGORITHM */

	feed_forward(input);
										/* 2. BACK-PROPAGATION ALGORITHM (feat.Sigmoid) */	

	//////////////////////////////////////  OUTPUT TO LAST HIDDEN LAYER  ////////////////////////////////////////////////////

	// 2-(1) Output layer to previous layer
	for (int i = 0; i<output_layer.cur_dim; i++)
	{
		output = output_layer.neuron[i]->activated; 	
		loss_function += (real_value[i] - output) * (real_value[i] - output);
		local_error = (real_value[i] - output) * output * (1 - output); // local error of ith neuron of the output layer

		/* Update bias(= b = w_0) */
		output_layer.neuron[i]->w_b += learning_rate * local_error * output_layer.neuron[i]->b;

		/* Update the weights which is connected to ith output neuron (From previous node 'j' to output node 'i') */
		for (int j = 0; j<output_layer.prev_dim; j++)
		{
			// Update the weight values by the amount of delta value(= learning rate X local_error X jth prev_layer neuron)
			output_layer.neuron[i]->w[j] += learning_rate*local_error*output_layer.prev_layer[j];;															  
			sum_transmited += local_error*output_layer.neuron[i]->w[j]; // For Next Layer, we need this "sum_transmitted" variable
		}
	}

	//////////////////////////////////////  HIDDEN TO HIDDEN LAYER  ////////////////////////////////////////////////////////////////////////////

	// 2-(2) hidden[i] to hidden[i-1] layer
	for (int i = (num_hidden_layer - 1); i >= 0; i--)
	{
		for (int j = 0; j<hidden_layer[i]->cur_dim; j++)
		{
			output = hidden_layer[i]->neuron[j]->activated;
			local_error = sum_transmited * output * (1 - output);

			/* Update the bias_weight */
			hidden_layer[i]->neuron[j]->w_b += learning_rate * local_error * hidden_layer[i]->neuron[j]->b;

			/* update the weight matrix of ith hidden layer (Weight From previous layer k to current layer j) */
			for (int k = 0; k < hidden_layer[i]->prev_dim; k++)
			{
				hidden_layer[i]->neuron[j]->w[k] += learning_rate*local_error*hidden_layer[i]->prev_layer[k];
				sum_of_layer += local_error * hidden_layer[i]->neuron[j]->w[k];  // To pass this sum for the next layer
			}
		}
		sum_transmited = sum_of_layer; // will be transmitted to the next layer(hidden layer[i-1])
		sum_of_layer = 0;			  // reset to 0 to calculate the new layer's error later
	}

	//////////////////////////////////////  FIRST HIDDEN TO INPUT LAYER  ////////////////////////////////////////////////////////////////////////////

	// 2-(3) first hidden layer to input layer
	for (int i = 0; i<input_layer.cur_dim; i++)
	{
		output = input_layer.neuron[i]->activated;
		local_error = sum_transmited * output * (1 - output);

		/* Update the bias_weight */
		input_layer.neuron[i]->w_b += learning_rate * local_error * input_layer.neuron[i]->b;

		/* update the weight matrix of input layer */
		for (int j = 0; j < input_layer.prev_dim; j++)
		{
			input_layer.neuron[i]->w[j] += learning_rate*local_error*input_layer.prev_layer[j];
			//printf("Updated Input Layer ith node to jth node Weights: %f\n", input_layer.neuron[i]->w[j]);
		}
	}

	//return the loss function(= cost, error)
	return loss_function / 2;
}


