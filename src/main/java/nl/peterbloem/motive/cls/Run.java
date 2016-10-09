package nl.peterbloem.motive.cls;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nodes.data.RDF;

import nl.peterbloem.kit.Global;

public class Run 
{

	
	
	@Option(name="--file", usage="This should be an RDF file, the type is detected by extension.")
	private static File file;
	
	@Option(name="--task", usage="TSV file containing the classification experiment.")
	private static File classTSV;
	
	@Option(
			name="--hubs",
			usage="Number of hubs to remove from the data (the more hubs removed, the smaller the instances become.")
	private static int classHubs = 0;
	
	@Option(
			name="--orcaSamples",
			usage="Number of samples from the null model in the FANMOD experiment.")
	private static int classFMSamples = 1000;
	
	@Option(
			name="--motiveSamples",
			usage="Number of subgraphs to sample in the motive experiment.")
	private static int classMotiveSamples = 1000000;
	
	@Option(
			name="--depth",
			usage="Depth to which to extract the instances.")
	private static int classDepth = 2;
	
	@Option(
			name="--mixingTime",
			usage="Mixing time for the curveball sampling algorithm (ie. the number of steps taken in the markov chain for each sample).")
	private static int classMixingTime = 10000;
	
	@Option(
			name="--numInstances",
			usage="The number of instances to use (samples from the total available)")
	private static int classNumInstances = 100;
	
	@Option(name="--help", usage="Print usage information.", aliases={"-h"}, help=true)
	private static boolean help = false;
	
	
	/**
	 * Main executable function
	 * @param args
	 */
	public static void main(String[] args) 
	{	
		Run run = new Run();
		
		// * Parse the command-line arguments
    	CmdLineParser parser = new CmdLineParser(run);
    	try
		{
			parser.parseArgument(args);
		} catch (CmdLineException e)
		{
	    	System.err.println(e.getMessage());
	        System.err.println("java -jar motive.jar [options...]");
	        parser.printUsage(System.err);
	        
	        System.exit(1);	
	    }
    	
    	if(help)
    	{
	        parser.printUsage(System.out);
	        
	        System.exit(0);	
    	}
    	
		Global.log().info("Using " + Global.numThreads() + " concurrent threads");
    	
  
		ClassExperiment exp = new ClassExperiment();
		
		try {
			exp.graph = RDF.readSimple(file);
		} catch (IOException e) {
			throw new RuntimeException("Could not read RDF input file.", e);
		}
		
		try {
			exp.map = ClassExperiment.tsv(classTSV);
		} catch (IOException e) {
			throw new RuntimeException("Could not read TSV classification file.", e);
		}	
		
		exp.seed = new Random().nextInt(100000);
		exp.hubsToRemove = classHubs;
		exp.samples = classFMSamples;
		exp.motiveSamples = classMotiveSamples;
		exp.instanceDepth = classDepth;
		exp.mixingTime = classMixingTime;
		exp.numInstances = classNumInstances;
		
		try {
			exp.main();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		    		
	}

}
