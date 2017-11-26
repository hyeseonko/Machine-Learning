#ifndef _MLP_H__
#define _MLP_H__

/////////////////////////////////////////////////////////////////////
//  Three Structures: Node ¡æ Layer ¡æ Multi-layer Perceptron(MLP)   //
/////////////////////////////////////////////////////////////////////

struct node {
	float b;	// bias
	float w_b;	// bias weight
	float *w;	// weight[]:= {wj1, wj2, wj3, wj4, ..., wjn}
	float activated; // Value After using activation function

	node();		// Constructor
	~node();	// Destructor
	void init(int prev_dim); // Initialise
};

struct layer {
	int cur_dim;    // the number of nodes in the layer
	node **neuron;
	int prev_dim;
	float *prev_layer;

	layer();	// Constructor
//	~layer();	// Destructor
	void init(int before_dim, int now_dim);
	void activation();  // sigmoid, ReLU, LeakyReLU
};

struct mlp {
	layer input_layer;
	layer **hidden_layer;
	layer output_layer;
	int num_hidden_layer;

	mlp();
//	~mlp();
	void init(int input_dim, int input_neuron, int hidden_dim[], int hidden_cnt, int output_dim);
	void connection_update(int what_layer);
	void feed_forward(float input[]);
	float train(float input[], float real_value[], float learning_rate);
	layer &getOutput()
	{
		return output_layer;
	}
};




#endif