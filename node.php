<?
/*
very simple php class for node in neural network - output based on sigmoid function
*/

Class Node{
	public $inputs,$weights, $bias;
	public $output;

	function Node(){
		unset($this->output);
		$this->bias=0;
	}

	function setInputs($input){
		$this->inputs = $input;
	}

	function setWeights($weight){
		$this->weights = $weight;
	}

	public function calcOutput(){
		$output=NULL;

		foreach($this->inputs as $key => $input){
			$output+=($input * $this->weights[$key]);
		}
		$output+=$this->bias;
		$this->output=(1/(1+exp(-1 * $output)));
	}
	
	function getOutput(){
		if(!isset($this->output)){
			$this->calcOutput();
		}
		return($this->output);
	}

}
?>
