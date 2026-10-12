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

public class MonthlyTrend {
    
    public static class MonthMapper extends Mapper<LongWritable, Text, Text, LongWritable> {
        private Text monthKey = new Text();
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
                
                String sourceName = parts[0].trim();
                String[] dateParts = sourceName.split("-");
                
                if (dateParts.length >= 5) {
                    String year = dateParts[3];
                    String month = dateParts[4];
                    String monthStr = year + "-" + month;
                    
                    long streams = Long.parseLong(parts[9].trim());
                    
                    monthKey.set(monthStr);
                    streamsValue.set(streams);
                    context.write(monthKey, streamsValue);
                }
                
            } catch (Exception e) {
                // 跳过解析失败的行
            }
        }
    }
    
    public static class MonthReducer extends Reducer<Text, LongWritable, Text, LongWritable> {
        public void reduce(Text key, Iterable<LongWritable> values, Context context) 
                throws IOException, InterruptedException {
            long sum = 0;
            for (LongWritable val : values) {
                sum += val.get();
            }
            context.write(key, new LongWritable(sum));
        }
    }
    
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Monthly Trend");
        job.setJarByClass(MonthlyTrend.class);
        job.setMapperClass(MonthMapper.class);
        job.setReducerClass(MonthReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(LongWritable.class);
        
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
