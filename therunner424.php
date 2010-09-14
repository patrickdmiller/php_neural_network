<?
include_once('./network.php');
$debug=0;
$myNetwork = new NeuralNetwork();
$myNetwork->initialize(4,2,4);
//$myNetwork->initializeFromFile('training2');
$myNetwork->debug=$debug;
$myNetwork->inputLayer(array(1,1,0,0));
$myNetwork->randomWeights();
$myNetwork->output();

$tester=array(
	array(1,0,0,0),
	array(0,1,0,0),
	array(0,0,1,0),
array(0,0,1,1)

);

//$tester=array(array(0,1,0),array(0,0,1),array(1,0,0));
for($iterations=0; $iterations<10000;$iterations++){
	foreach($tester as $test){
		$myNetwork->inputLayer($test);
		$myNetwork->adjust($test,.15);
	}
	if($iterations % 1000 ==0){
		echo "--".$iterations;
	}
}


foreach($tester as $test){
	$myNetwork->inputLayer($test);
	$myNetwork->output();
	echo "\n";
	foreach($myNetwork->getInputLayer() as $o)
		printf("\t %01.4f", $o);
	echo " |";
	foreach($myNetwork->getOutputLayer() as $o)
		printf("\t %01.4f", $o);
	echo " |";
	foreach($myNetwork->getHiddenLayer() as $o)
		printf("\t %01.4f", $o);
}



$tester=array(
	array(0,0,0,0,0,0,0),
	array(0,0,0,0,0,0,1),
	array(0,0,0,0,0,1,0),

);


for($iterations=0; $iterations<0;$iterations++){
	foreach($tester as $test){
		$myNetwork->inputLayer($test);
		$myNetwork->adjust($test,.15);
	}
	if($iterations % 5000 ==0){
		echo "--".$iterations;
		//$myNetwork->writeToFile('training23');
	}
}
echo "\n-";
foreach($tester as $test){
	$myNetwork->inputLayer($test);
	$myNetwork->output();
	echo "\n";
	foreach($myNetwork->getInputLayer() as $o)
		printf("\t %01.4f", $o);
	echo " |";
	foreach($myNetwork->getOutputLayer() as $o)
		printf("\t %01.4f", $o);
	echo " |";
	foreach($myNetwork->getHiddenLayer() as $o)
		printf("\t %01.4f", $o);
}

if($debug==1)
	print_r($myNetwork);

$myNetwork->writeToFile('training424');

?>
