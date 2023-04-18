function download_data(URL, input_file)
  
  if  ~exist(input_file, 'file')

    % this fails on some machines ubuntu 18.04
    if ispc
        system(sprintf('curl -L --output %s %s', input_file, URL))
    
    else
        try
          urlwrite(URL, input_file);
        catch
          this_dir = fileparts(mfilename('fullfile'));          
          if exist(fullfile(this_dir, 'download'), 'file')
            % remove eventual previously incomplete downloads
            delete(fullfile(this_dir, 'download'))
          end
          % if fails, may need to install wget e.g. via "brew install wget"
          % and may need to restart matlab following install
          system(sprintf('wget --verbose %s', URL)); 
          movefile(fullfile(this_dir, 'download'), input_file);
        end
    
    end

  end
  
end