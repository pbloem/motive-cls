package nl.peterbloem.motive.cls;

import static java.util.concurrent.TimeUnit.MINUTES;
import static nl.peterbloem.kit.Functions.dot;
import static nl.peterbloem.kit.Functions.subset;
import static nl.peterbloem.kit.Functions.tic;
import static nl.peterbloem.kit.Functions.toc;
import static nl.peterbloem.kit.Series.series;
import static org.nodes.LightUGraph.copy;
import static org.nodes.models.USequenceEstimator.perturbation;

import static nl.peterbloem.kit.Functions.tic;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.xerces.util.SynchronizedSymbolTable;
import org.nodes.DGraph;
import org.nodes.Graph;
import org.nodes.Graphs;
import org.nodes.LightUGraph;
import org.nodes.MapDTGraph;
import org.nodes.MapUTGraph;
import org.nodes.Subgraph;
import org.nodes.UGraph;
import org.nodes.ULink;
import org.nodes.UNode;
import org.nodes.UTGraph;
import org.nodes.algorithms.Nauty;
import org.nodes.data.Examples;
import org.nodes.data.RDF;
import org.nodes.models.EdgeListModel;
import org.nodes.models.USequenceEstimator;
import org.nodes.models.DegreeSequenceModel.Prior;
import org.nodes.motifs.AllSubgraphs;
import org.omg.Messaging.SyncScopeHelper;
import org.openrdf.rio.RDFFormat;

import nl.peterbloem.kit.FileIO;
import nl.peterbloem.kit.FrequencyModel;
import nl.peterbloem.kit.Functions;
import nl.peterbloem.kit.Generator;
import nl.peterbloem.kit.Global;
import nl.peterbloem.kit.MaxObserver;
import nl.peterbloem.kit.Series;
import nl.peterbloem.kit.data.Point;
import nl.peterbloem.kit.data.classification.Classification;
import nl.peterbloem.kit.data.classification.Classified;
import nl.peterbloem.motive.MotifSearchModel;
import nl.peterbloem.motive.UPlainMotifExtractor;
import orca.Orca;

public class ClassExperiment 
{
	/**
	 * How many samples to take from the FANMOD null model 
	 */
	public int samples = 1000;
	
	/**
	 * How often to iterate the null model Markov chain for each sample
	 */
	public int mixingTime = 10000;
	
	/**
	 * The significance level (for both fanmod and motive)
	 */
	public double alpha = 0.05;
	
	/**
	 * How many samples to take in the motive version
	 */
	public int motiveSamples = 100000;
	
	/**
	 * How many hubs to remove from the data before counting instances
	 * (this reduces the size of the instances)
	 */
	public int hubsToRemove = 3000;
	
	/**
	 * The number of instances to use (a sample form the total number available)
	 */
	public int numInstances = 100;
	
	/**
	 * How many steps to use for the instances
	 */
	public int instanceDepth = 2;
	
	/**
	 * The seed for the random number generator
	 */
	public int seed = 0; 
	
	public Comparator<String> natural = Functions.natural();
	
	public UGraph<String> graph = null;
	public Map<String, String> map = null;
	
	
	public void main()
		throws IOException
	{
		tic();
		
		// * Remove hubs
		LinkedList<UNode<String>> nodes = new LinkedList<UNode<String>>(graph.nodes());
		Comparator<UNode<String>> comp = new Comparator<UNode<String>>() 
		{
			@Override
			public int compare(UNode<String> n1, UNode<String> n2) 
			{
				int d1 = n1.degree(), 
				    d2 = n2.degree();
				
				return - Integer.compare(d1, d2);
			}
		};
		
		Collections.sort(nodes, comp);
		
		List<Integer> toRemove = new ArrayList<Integer>(hubsToRemove);
		while(toRemove.size() < hubsToRemove)
		{
			UNode<String> node = nodes.pop();
			
			if(! map.containsKey(node.label()))
				toRemove.add(node.index());
		}
		
		Comparator<Integer> intComp = Collections.reverseOrder();
		Collections.sort(toRemove, intComp);
				
		for(int index : toRemove)
			graph.get(index).remove();
		
		System.out.println(MaxObserver.quickSelect(10, Graphs.degrees(graph), intComp, false));
		
		System.out.println("graph read: " + Functions.toc() + " s, size " + graph.size());
		
		List<String> classInts = new ArrayList<String>(new LinkedHashSet<String>(map.values()));
		System.out.println("table read: " + map.size());

		int c = 0;
		Classified<UGraph<String>> data = Classification.empty();
		
		
		tic();
		// - Sample random instances
		for(String name : subset(map.keySet(), numInstances))
		{
			UNode<String> instanceNode = graph.node(name);
			UGraph<String> instance = instance(instanceNode, instanceDepth);
		
			data.add(instance, classInts.indexOf( map.get(name)));
		}
		
		double size = 0;
		double numLinks = 0;
		for(int i : series(data.size()))
		{
			size += data.get(i).size();
			numLinks += data.get(i).numLinks();
			
			dot(i, data.size());
		}
		
		size /= data.size();
		numLinks /= data.size();
		
		System.out.println("Instances loaded, n: " + size + ", m: " + numLinks + ", time: " + toc() + "s.");
		
		// * Collect all connected motifs (up to isomorphism) for the given sizes
		List<UGraph<String>> motifs = new ArrayList<UGraph<String>>();
		for(int msize : Arrays.asList(3, 4, 5))
			motifs.addAll(Graphs.allIsoConnected(msize, "x"));
		
		System.out.println(motifs.size() + " features");
		
		BufferedWriter motiveWriter = new BufferedWriter(new FileWriter(
				new File(String.format("motive.%05d.csv", seed))));
		BufferedWriter fanmodWriter = new BufferedWriter(new FileWriter(
				new File(String.format("fanmod.%05d.csv", seed))));
		
		for(int i : Series.series(data.size()))
		{
			Global.log().info("Starting instance " + i + " of " + data.size());
			
			UGraph<String> bg = Graphs.blank(data.get(i), "x");
			int cls = data.cls(i);
			
			motiveWriter.write(cls + "");
			for(double v : featuresMotive(bg, motifs))
				motiveWriter.write(", " + v);
			motiveWriter.newLine();
			motiveWriter.flush();
			
			fanmodWriter.write(cls + "");
			for(double v : featuresFANMOD(bg, motifs))
				fanmodWriter.write(", " + v);
			fanmodWriter.newLine();
			fanmodWriter.flush();
		}
		
		motiveWriter.close();
		fanmodWriter.close();
		
		try {
			FileIO.python(new File("."), "scripts/plot.classification.py");
		} catch (Exception e)
		{
			Global.log().warning("Failed to run plot script. The script has been copied to the output directory.  (trace:" + e + ")");
		}
	}

	private UGraph<String> instance(UNode<String> instanceNode, int depth) 
	{
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		
		UGraph<String> graph = instanceNode.graph();
		UGraph<String> instance = new LightUGraph<String>();
		
		Set<Integer> added = new LinkedHashSet<Integer>();
		
		UNode<String> iiNode = instance.add(instanceNode.label());
		map.put(instanceNode.index(), iiNode.index());
		added.add(instanceNode.index());
		
		expand(graph, instance, map, added, depth);
		
		return instance;
	}

	/**
	 * Expands 'instance', a subgraph of 'graph', by adding all neighbours of 
	 * nodes indicated in 'addded'. Repeats recursively to the given depth.
	 * 
	 * @param graph
	 * @param instance
	 * @param added Indices in the supergraph of nodes that were added in the last iteration.
	 */
	private void expand(UGraph<String> graph, UGraph<String> instance, Map<Integer, Integer> map, Set<Integer> added, int depth) 
	{
		if(depth == 0)
			return;
		
		Set<Integer> inInstance = new HashSet<Integer>(map.keySet());
		
		Set<Integer> newAdded = new LinkedHashSet<Integer>();
		for (int index : added)
			for (UNode<String> neighbor : graph.get(index).neighbors())
			{
				if (inInstance.contains(neighbor.index()))
				{					
					// If this happens, all relevant links should already be included
				}
				else if (map.containsKey(neighbor.index())) // node already added
				{
					if(! instance.get(map.get(index)).connected(instance.get(map.get(neighbor.index()))))
						instance.get(map.get(index)).connect(instance.get(map.get(neighbor.index())));
				} else // new node
				{
					UNode<String> nwNode = instance.add(neighbor.label());
					map.put(neighbor.index(), nwNode.index());
					
					instance.get(map.get(index)).connect(nwNode);
					
					newAdded.add(neighbor.index());
				}
			}
		
		// * Add all links between nodes in nwAdded	
		for(int oldIndex : newAdded)
			for(UNode<?> neighbor : graph.get(oldIndex).neighbors())
			{
				int oldNIndex = neighbor.index();	
				if(newAdded.contains(oldNIndex) && oldNIndex > oldIndex)
				{
					int index = map.get(oldIndex);
					int nIndex = map.get(oldNIndex);
					if(! instance.get(index).connected(instance.get(nIndex)))
						instance.get(index).connect(instance.get(nIndex));
				}
			}
		
		expand(graph, instance, map, newAdded, depth -1);
	}

	
	private List<Double> featuresFANMOD(final UGraph<String> graph, final List<UGraph<String>> motifs) 
	{
		FrequencyModel<UGraph<String>> dataCounts = count(graph, motifs);
		
		Global.log().info("Data count completed");
		
		// * Synched object to collect the counts from the different threads
		class Collector 
		{
			private Map<UGraph<String>, List<Integer>> nullCounts = new LinkedHashMap<>();
			/**
			 * We need a local (deep) copy of the motifs, or we get race conditions
			 * when the hashcodes are computed concurrently (in the collector 
			 * and in the threads)
			 */
			private List<UGraph<String>> local = new ArrayList<>();

			public Collector()
			{
				for(UGraph<String> motif : motifs)
				{
					UGraph<String> copy = MapUTGraph.copy(motif);
					
					local.add(copy);
					nullCounts.put(copy, new ArrayList<Integer>(samples));
				}
			}
			
			public synchronized void register(FrequencyModel<UGraph<String>> sampleCounts)
			{
				for(UGraph<String> token : local)
				{
					int count = (int)sampleCounts.frequency(token);
					List<Integer> list = nullCounts.get(token);
					list.add(count);
					nullCounts.put(token, list);
				}
			}
			
			public Map<UGraph<String>, List<Integer>> nullCounts()
			{
				return nullCounts;
			}
		}
	
		final Collector collector = new Collector();
			
		final USequenceEstimator<String> model = new USequenceEstimator<String>(graph, "x");
		
        ExecutorService executor = Executors.newFixedThreadPool(Global.numThreads());
        final AtomicInteger finished = new AtomicInteger(0); 

        class MThread extends Thread 
        {
        	private final List<UGraph<String>> local = new ArrayList<>();
        	
        	public MThread()
        	{	
				for(UGraph<String> motif : motifs)
					local.add(MapUTGraph.copy(motif));
        	}
				
			public void run()
			{	
				// * Generate a random graph
				final Generator<UGraph<String>> gen = model.uniform(mixingTime);
				UGraph<String> sample = gen.generate();
				
				// * Count its subgraphs
				FrequencyModel<UGraph<String>> sampleCounts = count(sample, motifs);
				
				// * Add the counts to the nullCounts
				collector.register(sampleCounts);
				
				finished.incrementAndGet();
			}
        }
        
		for(int i : series(samples))
			executor.execute(new MThread());
		
		// * Execute all threads and wait until finished
		executor.shutdown();
		try 
		{
			while(! executor.awaitTermination(1, MINUTES))
				Global.log().info(finished + " of " + samples + " completed.");
				
		} catch (InterruptedException e) {
			throw new RuntimeException(e);
		}
		
		Global.log().info("Sampling finished");
		
		Map<UGraph<String>, List<Integer>> nullCounts = collector.nullCounts();
		
		for(List<Integer> counts : nullCounts.values())
			Collections.sort(counts);
		
		List<Double> features = new ArrayList<Double>(motifs.size());
		for(UGraph<String> motif : motifs)
		{
			int dc = (int) dataCounts.frequency(motif);
			List<Integer> nc = collector.nullCounts().get(motif);
			
			features.add(prop(nc, dc) <= alpha ? 1.0 : 0.0);
		}
		
		return features;
	}
	
	private List<Double> featuresMotive(UGraph<String> graph, List<UGraph<String>> motifs) 
	{
		
		System.out.println("n: " + graph.size() + " m: " + graph.numLinks() + "");
		UPlainMotifExtractor<String> ex = new UPlainMotifExtractor<String>(graph, motiveSamples, 3, 5);
	
		double threshold = - Functions.log2(alpha);
		System.out.println("Threshold: " + threshold);
		
		double baseline = new EdgeListModel(Prior.COMPLETE).codelength(graph);
		System.out.println("Baseline: " + baseline);

		List<Integer> degrees = Graphs.degrees(graph);
				
		List<Double> features = new ArrayList<Double>(motifs.size());
		for(UGraph<String> motif : motifs)
		{
			List<List<Integer>> occ = ex.occurrences(motif);
			if(occ == null)
				occ = Collections.emptyList();
			
			double length = MotifSearchModel.sizeELInst(graph, degrees, motif, occ, true, -1);
			
			features.add( (baseline - length) > threshold ? 1.0 : 0.0 );
		}
	
		return features;
	}
	
	/**
	 * Returns the proportion of 'values' that is larger than or equal to 
	 * the given threshold.
	 * 
	 * @param values
	 * @param threshold
	 * @return
	 */
	private static double prop(List<Integer> values, int threshold)
	{
		int total = 0;
		for(int value : values)
		{
			if(value >= threshold)
				break;
			
			total ++;
		}
		
		return (values.size() - total) / (double) values.size();
	}
	
	private static FrequencyModel<UGraph<String>> count(UGraph<String> graph, List<UGraph<String>> subs)
	{
		Comparator<String> natural = Functions.natural();
		FrequencyModel<UGraph<String>> fm = new FrequencyModel<UGraph<String>>();

		Orca orca = new Orca(graph, true);
		
		for(UGraph<String> sub : subs)
			fm.add(sub, orca.count(sub, true));
		
//		System.out.println(fm);
		
		return fm;
	}

	/**
	 * Returns a blanked graph with string labels
	 * @param graph
	 * @param label
	 * @return
	 */
	private static UGraph<String> str(UGraph<Integer> graph, String label) 
	{
		UGraph<String> result = new MapUTGraph<String, String>();

		for(UNode<Integer> node : graph.nodes())
			result.add(label);
		
		for(ULink<Integer> link : graph.links())
			result.get(link.first().index()).connect(
					result.get(link.second().index()));
		
		return result;
	}

	/**
	 * Does a run of the curveball algorithm for each graph and dumps the 
	 * perturbation scores to a CSV file.
	 * 
	 * @param graphs
	 */
	private static <L> void estimateMixingTime(Classified<UGraph<L>> graphs)
		throws IOException
	{
		int n = 100000; int m = 3;
		List<List<Double>> scores = new ArrayList<List<Double>>();
		
		for(int c : Series.series(m))
		{
			UGraph<L> graph = Functions.choose(graphs);
			
			List<Double> s = new ArrayList<Double>(n);
			
			List<Set<Integer>> 
				start = USequenceEstimator.adjacencies(graph),
	            current = USequenceEstimator.adjacencies(graph);
			
			for(int i : series(n))
			{
				USequenceEstimator.step(current);
				s.add(perturbation(start, current));
			}
			
			scores.add(s);
			
			dot(c, graphs.size());
		}
		
		// * Write to CSV
		
		BufferedWriter writer = new BufferedWriter(new FileWriter(new File("./am.csv")));
		for(int row : series(n))
		{
			for(int col : series(m))
				writer.write((col != 0? ", " : "") +  scores.get(col).get(row));
			
			writer.write("\n");
		}
		writer.close();
		
	}
	
	public static Map<String, String> tsv(File file)
		throws IOException
	{
		Map<String, String> map = new LinkedHashMap<String, String>();
		
		BufferedReader reader = new BufferedReader(new FileReader(file));
		
		reader.readLine(); // skip titles
		String line = reader.readLine();
		
		while(line != null)
		{
			String[] split = line.split("\\s");
			map.put(split[1], split[2]);
			
			line = reader.readLine();
		}
		
		return map;
	}
}
