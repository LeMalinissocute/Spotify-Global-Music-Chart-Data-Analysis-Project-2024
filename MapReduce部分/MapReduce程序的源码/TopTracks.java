package com.spotify;

import com.opencsv.CSVParser;
import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import java.io.IOException;
import java.io.StringReader;
import java.util.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class TopTracks {
    
    public static class TrackMapper extends Mapper<LongWritable, Text, Text, LongWritable> {
        private Text trackKey = new Text();
        private LongWritable streamsValue = new LongWritable();
        private CSVParser parser = new CSVParserBuilder()
                .withSeparator(',')
                .withQuoteChar('"')
                .withEscapeChar('\\')
                .build();
        
        public void map(LongWritable key, Text value, Context context) 
                throws IOException, InterruptedException {
            String line = value.toString();
            
            if (line.startsWith("Source.Name")) {
                return;
            }
            
            try {
                CSVReader reader = new CSVReaderBuilder(new StringReader(line))
                        .withCSVParser(parser)
                        .build();
                String[] parts = reader.readNext();
                reader.close();
                
                if (parts == null || parts.length < 10) {
                    return;
                }
                
                String trackName = parts[4].trim();
                String artistName = parts[3].trim();
                String trackKeyStr = trackName + "|" + artistName;
                long streams = Long.parseLong(parts[9].trim());
                
                trackKey.set(trackKeyStr);
                streamsValue.set(streams);
                context.write(trackKey, streamsValue);
                
            } catch (Exception e) {
                // 跳过解析失败的行
            }
        }
    }
    
    public static class TrackReducer extends Reducer<Text, LongWritable, Text, LongWritable> {
        private TreeMap<Long, String> topMap = new TreeMap<>();
        
        public void reduce(Text key, Iterable<LongWritable> values, Context context) 
                throws IOException, InterruptedException {
            long sum = 0;
            for (LongWritable val : values) {
                sum += val.get();
            }
            topMap.put(sum, key.toString());
            if (topMap.size() > 10) {
                topMap.remove(topMap.firstKey());
            }
        }
        
        protected void cleanup(Context context) 
                throws IOException, InterruptedException {
            for (Map.Entry<Long, String> entry : topMap.descendingMap().entrySet()) {
                context.write(new Text(entry.getValue()), new LongWritable(entry.getKey()));
            }
        }
    }
    
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Top Tracks");
        job.setJarByClass(TopTracks.class);
        job.setMapperClass(TrackMapper.class);
        job.setReducerClass(TrackReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(LongWritable.class);
        
        // 关键：args[0] = 输入路径，args[1] = 输出路径
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
