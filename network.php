<?	/*
	Patrick Miller
	neural network with back propagation
	*/

include_once('./node.php');

Class NeuralNetwork{
	public $nodes;

	//initialize takes comma separated list of layers ex: 3,2,3
	function initialize(){
		for ($i = 0; $i < func_num_args(); $i++) {
			for($j=0; $j< func_get_arg($i); $j++){
				$this->nodes[$i][$j] = new Node();
			}
		}
		
	}

	//will load from a file that the class has saved an object to (see writeToFile())
	function initializeFromFile($filename){
		$lines=file($filename);
		$structure=explode(',',$lines[0]);
		print_r($structure);
		for($i=0; $i<count($structure);$i++){
			for($j=0; $j< $structure[$i];$j++){
				$this->nodes[$i][$j]=new Node();		
			}					
		}
		for($i=1;$i<count($lines);$i++){
			$inw=explode('|',$lines[$i]);
			for($j=0;$j<count($inw);$j++){
				$this->nodes[$i][$j]->setWeights(explode(",",$inw[$j]));
			}
		}	
	}

	//set the input layer (takes an array)
	function inputLayer($input){
		for($i=0;$i<count($this->nodes[0]);$i++){
			if($this->debug==1)
				echo "\n setting 0 $i to ".$input[$i];
			$this->nodes[0][$i]->output=$input[$i];
		}
		if($this->debug==1)
			print_r($this->nodes);
	}

	//set random weights
	function randomWeights(){
		srand(time());
		for($i=1;$i<count($this->nodes);$i++){
			for($j=0;$j<count($this->nodes[$i]);$j++){
				$warray=NULL;
				for($w=0;$w<count($this->nodes[($i-1)]);$w++){
					$warray[$w]=((rand()%98)+1)/100;
				}
				//print_r($warray);	
				$this->nodes[$i][$j]->setWeights($warray);
				$this->nodes[$i][$j]->bias=((rand()%98)+1)/100;
			}
		}
	}

	//calculate the values of the nodes in the network from layer 0 to output
	function output(){
		for($i=1;$i<count($this->nodes);$i++){
			for($j=0;$j<count($this->nodes[$i]);$j++){
				$inarray=NULL;
				for($k=0;$k<count($this->nodes[$i-1]);$k++){
					if($this->debug==1)
						echo "\n--- getting output for ".($i-1)." $k";
					$inarray[$k]=$this->nodes[$i-1][$k]->getOutput();
					if($this->debug==1)
						echo "\n got output";
				}
				if($this->debug==1)
					echo "\n setting inputs for : $i $j";
				$this->nodes[$i][$j]->setInputs($inarray);
				$this->nodes[$i][$j]->calcOutput();
			}
		}
	}

	//single iteration of adjusting weights using back propagation
	function adjust($t=NULL, $n=.1){
		$this->output();
		//if no t then just make it train on input (encoder type)
		if($t==NULL)
				$t=$this->nodes[0];
		$max=count($this->nodes)-1;
			$tempWeights=NULL;
			//calc errors on output layer
			for($i=0; $i<count($this->nodes[$max]); $i++){
				$currentOut=$this->nodes[$max][$i]->getOutput();
				//calc error 
				$error[$max][$i]=$currentOut * (1-$currentOut)  * ($t[$i]-$currentOut);

				//adjust the bias (new b = old b + n(old b * error)
				$tempBias[$max][$i]=$this->nodes[$max][$i]->bias+($n * $error[$max][$i] * $this->nodes[$max][$i]->bias);
				if($this->debug==1)
					echo "\n adjusting bias [$max][$i] from ".$this->nodes[$max][$i]->bias." to ".$tempBias[$max][$i];
				//build temp weight array for each weight of this node
				if($this->debug==1)
					echo "\n error [".$max."][".$i."]=".$error[$max][$i];

				for($j=0;$j<count($this->nodes[$max][$i]->weights);$j++){
					if($this->debug==1)
						echo "\n newweight=".$this->nodes[$max][$i]->weights[$j]."+(".$n." * ".$error[$max][$i]." * ".$this->nodes[$max][$i]->inputs[$j].")";
					$tempWeights[$max][$i][$j]=$this->nodes[$max][$i]->weights[$j]+($n * $error[$max][$i] * $this->nodes[$max][$i]->inputs[$j]);
					
					if($this->debug==1)
						echo "\n adjusting".$this->nodes[$max][$i]->weights[$j]." to ".$tempWeights[$max][$i][$j];
				}
			}
			if($this->debug==1)
				echo "\n done setting weights on output layer";
			//adjust weights on hidden layers (max-1 to 1, 0 is input layer)
			for($layer=($max-1); $layer>0; $layer--){
				if($this->debug==1)
					echo "\ngoing into layer".$layer;
				for($node=0;$node<count($this->nodes[$layer]);$node++){
					$currentOut=$this->nodes[$layer][$node]->getOutput();
					//the sum of the errors from the next layer after this one
					$tempSum=0;
					for($nextnode=0;$nextnode<count($this->nodes[$layer+1]);$nextnode++){
						if($this->debug==1)
							echo "\nsumming up (".$error[$layer+1][$nextnode]."*".$this->nodes[$layer+1][$nextnode]->weights[$node].")";
						$tempSum+=($error[$layer+1][$nextnode]*$this->nodes[$layer+1][$nextnode]->weights[$node]);
					}
					$error[$layer][$node]= $currentOut * (1-$currentOut) * ($tempSum);
					if($this->debug==1)
						echo "\n error $layer $node = ".$error[$layer][$node];
					//adjust the bias
					$tempBias[$layer][$node]=$this->nodes[$layer][$node]->bias+($n * $error[$layer][$node] * $this->nodes[$layer][$node]->bias);
					if($this->debug==1)
						echo "\n adjusting bias [$layer][$node] from ".$this->nodes[$layer][$node]->bias." to ".$tempBias[$layer][$node];
					//adjust new weight
					for($j=0; $j<count($this->nodes[$layer-1]);$j++){
						if($this->debug==1)
							echo "\n newweight $layer | $node | $j=".$this->nodes[$layer][$node]->weights[$j]." + (".$n." * ".$error[$layer][$node]." * ".$this->nodes[$layer][$node]->inputs[$j].")";
						$tempWeights[$layer][$node][$j]=$this->nodes[$layer][$node]->weights[$j]+($n * $error[$layer][$node] * $this->nodes[$layer][$node]->inputs[$j]);
						if($this->debug==1)
							echo "\n adjusting".$this->nodes[$layer][$node]->weights[$j]." to ".$tempWeights[$layer][$node][$j];
					}
				}
			}
			if($this->debug==1)
				print_r($tempWeights);
			//apply weight and bias change to all the nodes (not input layer)
			for($i=1; $i<count($this->nodes);$i++){
				for($j=0;$j<count($this->nodes[$i]);$j++){
					
					if($this->debug==1)
						echo "\n setting weight and bias of node $i $j";
					$this->nodes[$i][$j]->setWeights($tempWeights[$i][$j]);
					//bias
					$this->nodes[$i][$j]->bias=$tempBias[$i][$j];
				}
			}
		//recalculate
		$this->output();
	}

	//returns input layer
	function getInputLayer(){
		$outarray=NULL;
		for($i=0;$i<count($this->nodes[0]);$i++){
			$outarray[$i]=$this->nodes[0][$i]->output;
		}
		return($outarray);
	}

	//returns output layer
	function getOutputLayer(){
		$max=count($this->nodes)-1;
		$outarray=NULL;
		for($i=0;$i<count($this->nodes[$max]);$i++){
			$outarray[$i]=$this->nodes[$max][$i]->output;
		}
		return($outarray);
	}

	//returns hidden layer just before output layer
	function getHiddenLayer(){
		$max=count($this->nodes)-2;
		$outarray=NULL;
		for($i=0;$i<count($this->nodes[$max]);$i++){
			$outarray[$i]=$this->nodes[$max][$i]->output;
		}
		return($outarray);
	}

	//writes network state to file
	function writeToFile($filename){
		$fh = fopen($filename,'w') or die("cant open");
		for ($i=0;$i<count($this->nodes);$i++){
			$data.=count($this->nodes[$i]).",";
		}
		$data=substr_replace($data,"\n",-1);
		
		for ($i=1;$i<count($this->nodes);$i++){
			for($j=0;$j<count($this->nodes[$i]);$j++){
				$out=$this->nodes[$i][$j]->weights;
				foreach($out as $o)
					$data.=$o.",";
				$data=substr_replace($data,"|",-1);
			}
			$data=substr_replace($data,"\n",-1);
		}
		fwrite($fh,$data);
		fclose($fh);
		echo "\n wrote to file".$filename;
	}
}
?>
