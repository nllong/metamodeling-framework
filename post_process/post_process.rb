require 'openstudio-analysis'
require 'optparse'
require 'date'

options = {
    server: 'http://localhost:3000',
    download: false,
    post_process: false,
    analysis_id: nil,
    rename_to: nil
}

parser = OptionParser.new do |opts|
  opts.banner = "Usage: post_process.rb [options]"

  opts.on('-s', '--server host', 'Server Host URL') do |server|
    options[:server] = server
  end

  opts.on('-a', '--analysis id', 'Analysis ID to Download or Directory Name to Post Process') do |id|
    options[:analysis_id] = id
  end

  opts.on('-r', '--rename-to name', 'Rename download directory to <name> (no spaces)') do |rename_to|
    options[:rename_to] = rename_to
  end

  opts.on('--download', 'Download Data') do
    options[:download] = true
  end

  opts.on('--post-process', 'Post Process Data') do
    options[:post_process] = true
  end
end
parser.parse!

unless options[:download] || options[:post_process]
  puts "Pass either --download or --post-process"
  exit
end

if options[:download] && !options[:analysis_id]
  puts "If --download, then must pass analysis_id to download (e.g. -a <id> --download)"
  exit
end

# These are hard coded for now. Ideally, this would be a passed file.
results_metadata = [
    {
        file: 'variables.json',
        data: [
            {level_1: '_id', rename_to: 'id', order: 1},
            {level_1: 'run_start_time', order: 1},
            {level_1: 'run_end_time', order: 2}
        ]
    },
    {
        file: 'results.json',
        data: [
            {
                level_1: 'ambient_loop_prototype_building_by_location',
                level_2: 'building_type',
                rename_to: '', # if there is no rename_to, then the name is set to the key
                order: 1 # if there are duplicates, then the fields will be sorted alphabetically
            },
            {
                level_1: 'ambient_loop_temperature_setpoint',
                level_2: 'design_delta',
                rename_to: '',
                order: 1
            },
            {
                level_1: 'ambient_loop_temperature_setpoint',
                level_2: 'setpoint_temperature',
                rename_to: '',
                order: 1
            },
            {
                level_1: 'internal_loads_multiplier',
                level_2: 'lpd_multiplier',
                rename_to: '',
                order: 1
            },
            {
                level_1: 'internal_loads_multiplier',
                level_2: 'epd_multiplier',
                rename_to: '',
                order: 1
            },
            {
                level_1: 'internal_loads_multiplier',
                level_2: 'lpd_average',
                rename_to: '',
                order: 1
            },
            {
                level_1: 'internal_loads_multiplier',
                level_2: 'epd_average',
                rename_to: '',
                order: 1
            },
            {
                level_1: 'set_window_to_wall_ratio',
                level_2: 'wwr',
                rename_to: 'wwr',
                order: 1
            },
            {
                level_1: 'set_roof_insulation_r_value',
                level_2: 'roof_r',
                rename_to: 'roof.r_value',
                order: 1
            },
            {
                level_1: 'set_exterior_wall_insulation_r_value',
                level_2: 'wall_r',
                rename_to: 'wall.r_value',
                order: 1
            },
            {
                level_1: 'set_heat_pump_heating_coil_rated_cop',
                level_2: 'heat_cop',
                rename_to: 'heat_pump.heating_cop',
                order: 1
            },
            {
                level_1: 'set_heat_pump_cooling_coil_rated_cop',
                level_2: 'cool_cop',
                rename_to: 'heat_pump.cooling_cop',
                order: 1
            },
            {
                level_1: 'set_schedule_profile_start_end_times',
                level_2: 'new_start',
                rename_to: 'schedule.start',
                order: 1
            },
            {
                level_1: 'set_schedule_profile_start_end_times',
                level_2: 'new_end',
                rename_to: 'schedule.end',
                order: 1
            },
            {
                level_1: 'openstudio_results',
                level_2: 'annual_peak_electric_demand',
                rename_to: '',
                order: 9
            },
            {
                level_1: 'openstudio_results',
                level_2: 'district_cooling_ip',
                rename_to: '',
                order: 9
            },
            {
                level_1: 'openstudio_results',
                level_2: 'district_heating_ip',
                rename_to: '',
                order: 9
            },
            {
                level_1: 'openstudio_results',
                level_2: 'electricity_ip',
                rename_to: '',
                order: 9
            },
            {
                level_1: 'openstudio_results',
                level_2: 'eui',
                rename_to: '',
                order: 9
            },
            {
                level_1: 'openstudio_results',
                level_2: 'total_site_eui',
                rename_to: '',
                order: 9
            },
            {
                level_1: 'openstudio_results',
                level_2: 'unmet_hours_during_occupied_cooling',
                rename_to: '',
                order: 9
            },
            {
                level_1: 'openstudio_results',
                level_2: 'unmet_hours_during_occupied_heating',
                rename_to: '',
                order: 9
            },
        ]
    }
]

def post_process_analysis_id(results_dir, results_metadata)
  if !Dir.exist?(results_dir)
    raise "Results directory does not exist, exiting: #{results_dir}"
    exit 1
  end

  # Go through the directories and update the reports to add in the last column of data.
  File.open("#{results_dir}/simulation_results.csv", 'w') do |new_file|
    Dir["#{results_dir}/*/*.csv"].each.with_index do |file, file_index|
      puts "Processing file #{file}"
      dir = File.dirname(file)
      new_header = []
      new_data = []

      results_metadata.each do |file|
        if File.exist? "#{dir}/#{file[:file]}"
          json = JSON.parse(File.read("#{dir}/#{file[:file]}"))
          metadata = file[:data].sort_by {|a| [a['order'], a['rename_to'], a['level_1'], a['level_2']]}

          metadata.each do |metadatum|
            # calculate the field name
            if metadatum[:rename_to] && !metadatum[:rename_to].empty?
              new_header << metadatum[:rename_to]
            elsif metadatum[:level_2] && !metadatum[:level_2].empty?
              new_header << "#{metadatum[:level_1]}.#{metadatum[:level_2]}"
            else
              new_header << metadatum[:level_1]
            end

            if metadatum[:level_2] && !metadatum[:level_2].empty?
	      if json[metadatum[:level_1]]
                new_data << json[metadatum[:level_1]][metadatum[:level_2]]
              else
	        puts "Could not find covariate #{metadatum[:level_1]}.#{metadatum[:level_2]}... will continue"
	      end
            else
              new_data << json[metadatum[:level_1]]
            end
          end
        end
      end

      # puts "New data are: #{new_header} : #{new_data}"
      File.readlines(file).each.with_index do |line, index|
        if file_index.zero? && index.zero?
          # write out the header into the new file
          new_file << "#{line.gsub(' ', '').chomp},#{new_header.join(',')}\n"
        elsif index.zero?
          # ignore the headers in the other files
          next
        else
          new_file << "#{line.chomp},#{new_data.join(',')}\n"
        end
      end
    end
  end

  # generate summary statistics on the simulations
  File.open("#{results_dir}/summary.csv", 'w') do |out_file|
    Dir["#{results_dir}/*/*.osw"].each.with_index do |file, file_index|
      headers = ["index", "uuid", "total_runtime", "energyplus_runtime", "total_measure_runtime", "other_runtime", "number_of_measures"]
      dir = File.dirname(file)
      json = JSON.parse(File.read(file))

      # The first file sets the length of the measures!
      if file_index.zero?
        json['steps'].each do |h|
          headers << "#{h['name']}_runtime"
        end
      end

      new_data = []
      new_data << file_index
      new_data << json['osd_id']
      start_time = DateTime.parse(json['started_at'])
      completed_at = DateTime.parse(json['completed_at'])
      total_time = completed_at.to_time - start_time.to_time
      new_data << total_time

      # read in the results.json
      json_file = "#{dir}/results.json"
      energyplus_runtime = 9999
      if File.exist? json_file
        results_json = JSON.parse(File.read(json_file))
        energyplus_runtime = results_json['ambient_loop_reports']['energyplus_runtime']
        new_data << energyplus_runtime
      else
        new_data << 'unknown ep runtime'
      end

      total_measure_time = 0
      measure_run_times = []
      json['steps'].each do |h|
        start_time = DateTime.parse(h['result']['started_at'])
        completed_at = DateTime.parse(h['result']['completed_at'])
        delta_time = completed_at.to_time - start_time.to_time
        total_measure_time += delta_time
        measure_run_times << delta_time
      end
      new_data << total_measure_time
      new_data << total_time - energyplus_runtime - total_measure_time
      new_data << json['steps'].size
      new_data += measure_run_times

      out_file << "#{headers.join(',')}\n" if file_index.zero?
      out_file << "#{new_data.join(',')}\n"
    end
  end
end

if options[:download]
  api = OpenStudio::Analysis::ServerApi.new(hostname: options[:server])
  if api.alive?
    base_dir = options[:rename_to].nil? ? options[:analysis_id] : options[:rename_to]
    if Dir.exist? base_dir
      warn "Directory exists #{base_dir}. Will continue to add results into this directory, make sure that the results are from the same analysis."
    else
      Dir.mkdir base_dir
    end

    puts "Downloading results for analysis id: #{options[:analysis_id]}"
    if ['completed', 'started'].include? api.get_analysis_status(options[:analysis_id], 'batch_run')
      # This API endpoint will only return the completed simulations so there is no need to check if they
      # are complete
      results = api.get_analysis_results(options[:analysis_id])
      results[:data].each do |dp|
        dir = "#{base_dir}/#{dp[:_id]}"
        # check if this has already been downloaded by looking for the directory
        if Dir.exist? dir
          puts "Simulation already downloaded"
        else
          puts "Saving results for simulation into directory: #{dir}"
          Dir.mkdir dir unless Dir.exist? dir

          # save off the JSON snippet into the new directory
          File.open("#{dir}/variables.json", 'w') {|f| f << JSON.pretty_generate(dp)}

          # grab the datapoint json, and save off the results (which contain some of the resulting covariates)
          results = api.get_datapoint(dp[:_id])
          File.open("#{dir}/results.json", 'w') {|f| f << JSON.pretty_generate(results[:data_point][:results])}

          # save off some of the results: timeseries, datapoint json, and out.osw
          api.download_datapoint_report(dp[:_id], 'ambient_loop_reports_report_timeseries.csv', dir)
          api.download_datapoint_report(dp[:_id], 'out.osw', dir)

          # Do not download the datapoint for now since it can be large
          # api.download_datapoint(dp[:_id], dir)
        end
      end
    else
      puts "Unknown status of analysis. Are you connecting to the right machine? Is the Analysis ID correct?"
    end
  else
    puts "Server is not running. Trying to process data using cached files"
  end
end

if options[:post_process]
  if options[:analysis_id]
    post_process_analysis_id(options[:analysis_id], results_metadata)
  else
    puts 'Must pass in path of folder (use -a <analysis_id> argument>)'
    exit 1
  end
end

