function download_data(URL, input_file)
  
  if  ~exist(input_file, 'file')

    % this fails on some machines ubuntu 18.04
    % system(sprintf('curl -L --output %s %s', input_file, URL))
    try
      urlwrite(URL, input_file);
    catch
      system(sprintf('wget --verbose %s', URL)); 
      movefile(fullfile(this_dir, 'download'), input_file);
    end

  end
  
end