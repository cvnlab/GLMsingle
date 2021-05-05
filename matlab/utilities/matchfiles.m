function f = matchfiles(patterns,sorttype)

% function f = matchfiles(patterns,sorttype)
%
% <patterns> is
%   (1) a string that matches zero or more files or directories (wildcards '*' okay)
%   (2) the empty matrix []
%   (3) a cell vector of zero or more things like (1) or (2)
% <sorttype> (optional) is how to sort in each INDIVIDUAL match attempt.
%   't' means sort by time (newest first)
%   'tr' means sort by time (oldest first)
%   default is [], which means to sort alphabetically by explicitly using MATLAB's sort function.
%   (note that MATLAB's sort function may sort differently than UNIX's ls function does!)
%   Note that the sorting does NOT break the order of cell vector elements; for example,
%   if A, B, or C are wildcards, then {A B C} still has the matched elements of A before
%   the matched elements of B, which are still before the matched elements of C.
%
% return a cell vector of strings containing paths to the matched files and/or directories.
% if there are no matches for an individual match attempt, we issue a warning.
%
% this function should be fully functional on Mac and Linux.  however, on Windows,
% we have the following limitations:
% - you cannot use the '?' operator
% - you can use the '*' operator only once and at the end of the expression
%   (not in an intermediate directory)
%
% on Mac and Linux, if we run into the too-many-files limitation of the ls command,
% we will resort to the alternative mode described above, and this inherits the 
% same limitations.
%
% history:
% 2019/09/23 - MAJOR BUG. The intention is that individual elements of a cell vector
%              get their own order in the results. We have fixed the code such that this is true.
% 2018/12/22 - we now return duplicates and issue a warning if duplicates are found.
% 2017/01/31 - switch to using Keith Jamison's fullfilematch.m implementation
% 2011/09/28 - if ls returns too many files, resort to alternative.  also, the alternative mode now allows sorttype to be specified.
% 2011/08/07 - allow empty matrix as an input
% 2011/04/02 - now, works on Windows (in a limited way)
% 2011/04/02 - oops, time-sorting behavior did not work.  bad bug!!!
% 2011/02/24 - escape spaces in patterns using \.  this fixes buggy behavior.
% 2011/01/21 - explicitly use MATLAB's sort function to ensure consistency across platforms.

% input
if ~exist('sorttype','var') || isempty(sorttype)
  sorttype = [];
end

% do it
f = fullfilematch(patterns,[],sorttype);
return;

%%%%%%%%%%%%%%%% CLONE OF fullfilematch.m from Keith Jamison, but with modifications

function files = fullfilematch(filestrings,case_sensitive,sorttype)

% function files = fullfilematch(filestrings,[case_sensitive=true],[sorttype=''])
%
% Find files with wildcard matching.
%
% Inputs:
%   filestrings: string or cell array of strings with path(s) to search for
%       - Paths can include ? or * wildcards anywhere in string
%   case_sensitive (optional): Use case sensitive search? default=true
%   sorttype (optional): '' = alphabetical (default)
%                        't' = newest->oldest
%                        'tr' = 'oldest->newest'
%                        
% Outputs:
%   files: cell array of (potentially non-unique) matching filenames
%
% Example: 
% > F=fullfilematch('~/somedir*/*.mat')
% F = 
%    '~/somedir/run1.mat'
%    '~/somedir/run2.mat'
%    '~/somedir/run3.mat'
%    '~/somedirA/run1.mat'
%    '~/somedirB/run1.mat'

% KJ Update 10/18/2016: Overhaul to allow wildcards in middle of path, and
%   to add sorting options (for use with cvnlab code)
% KJ Update 12/14/2016: Assume default directory='.' (pwd)

if(nargin < 2 || ~exist('case_sensitive','var') || isempty(case_sensitive))
    case_sensitive = true;
end

if(nargin < 3 || ~exist('sorttype','var') || isempty(sorttype))
    sorttype = '';
end

if(ischar(case_sensitive))
    if(strcmpi(case_sensitive,'ignorecase'))
        case_sensitive = false;
    else
        case_sensitive = true;
    end
end

if(isempty(filestrings))
    files = [];
    return;
end



% Handle the case of cell vectors up front in order to sure that order is preserved.
if iscell(filestrings)
  files = {};
  for p=1:length(filestrings)
    files = [files; fullfilematch(filestrings{p},case_sensitive,sorttype)];
  end
  return;
end




if(~iscell(filestrings))
    filestrings = {filestrings};
end

%make sure we can handle '\' filesep for Windows
if(isequal(filesep,'\'))
    fsep='[/\\]'; 
else
    fsep=filesep;
end


%% handle wildcards in the middle of path
%  eg: expand {'/data/experiment*/*.mat'} 
%   -> {'/data/experiment1/*.mat' 
%       '/data/experiment2/*.mat' 
%       '/data/experiment3/*.mat'}
filestrings0={};

for f = 1:numel(filestrings)
    filestr = filestrings{f};
    if(isdir(filestr) || ~any(ismember(filestr,'*?')))
        files_tmp = {filestr};
    else
        fparts=regexp(filestr,fsep,'split');

        if(numel(fparts)==1)
            %if no directory separators in input, pass directly to next
            %step (eg: input is '*' or '*.mat')
            files_tmp=fparts;
            filestrings0=[filestrings0; files_tmp(:)];
            continue;
        end
        
        %if first character is '/', keep a '/' at the beginning of the new
        %string
        if(~isempty(regexp(filestr(1),fsep))) %#ok<RGXP1>
            files_tmp={'/'};
        else
            files_tmp={''};
        end

        
        %loop through DIRECTORIES in path.  whenever we encounter a 
        % wildcard, call fullfilematch on the parent directory to find 
        % matching subdirectories, possibly returning multiple new
        % directories for the next level of the path (this is OK since
        % both fullfilematch() and strcat() can accept strings or cell
        % arrays of strings)
        
        for p = 1:numel(fparts)-1
            if(isempty(fparts{p}))
                continue;
            end
            if(any(ismember(fparts{p},'*?')))
                files_tmp=fullfilematch(strcat(files_tmp,fparts{p}),case_sensitive);
            else
                files_tmp=strcat(files_tmp,fparts{p});
            end
            if(isempty(files_tmp))
                break;
            end
            files_tmp=strcat(files_tmp,'/');
        end
        if(~isempty(files_tmp))
            %prune final list to only include directories, then tack on the 
            % filename part (which may include wildcards) to all, before
            % continuing on to the file-name wildcard search
            files_tmp=files_tmp(cellfun(@isdir,files_tmp));
            files_tmp=strcat(files_tmp,fparts{end});
        end
    end
    filestrings0=[filestrings0; files_tmp(:)];
end
% new filestrings is a cell array that may include many more entries than 
% the input if there were directory wildcards
filestrings=filestrings0;

%% main filename wildcard matching for each filestring
%   (only operates on the last path element., ie: the file name)
%  eg: expand {'/data/experiment/*.mat'}
%   -> {'/data/experiment/run1.mat'
%       '/data/experiment/run2.mat'}

files = {};
filedates = {};
for f = 1:numel(filestrings)
    [files_tmp,filedates_tmp] = aux_fullfilematch(filestrings{f},case_sensitive);
    files=[files(:); files_tmp(:)];
    filedates=[filedates(:); filedates_tmp(:)];
end

% check for duplicate filenames (if there are, issue a warning)
[~,iu] = unique(files);
if length(iu) < length(files)
  warning('Duplicate filenames found. We are including the duplicates! Consider using unique.m on the output.');
  %files=files(iu);
  %filedates=filedates(iu);
end

% sort by filename or by date
if strcmpi(sorttype,'t')
  [~,ii] = sort(cat(2,filedates{:}),2,'descend');
elseif strcmpi(sorttype,'tr')
  [~,ii] = sort(cat(2,filedates{:}));
elseif strcmpi(sorttype,'none')
    ii=1:numel(files);
else
  [~,ii] = sort(cat(2,files));
end

files = files(ii);

%% helper function that does the work to match individual filestrings
% returns filenames and dates to allow date-sorting in main function
function [files,filedates] = aux_fullfilematch(filestr,case_sensitive)
if(isdir(filestr))
    files = {filestr};
    filestruct=dir(filestr);
    %pretty sure '.' is always first, but just in case....
    i=find(strcmp({filestruct.name},'.'),1,'first');
    filedates=filestruct(i).datenum;
    return;
end
    
[filedir,fpattern,fext] = fileparts(filestr);
fpattern = strrep([fpattern fext],'*','.*');
fpattern = strrep(fpattern,'?','.');
fpattern = strrep(fpattern,'(','\(');
fpattern = strrep(fpattern,')','\)');
fpattern = strrep(fpattern,'+','\+');
fpattern = ['^' fpattern '$'];

removeprefix='';
if(isempty(filedir))
    filedir='.';
    removeprefix='./';
end
filestruct = dir(filedir);
if(numel(filestruct) == 1 && filestruct(1).isdir)
    [filedir2,~,~] = fileparts(filedir);
    if(filedir2(end)~='/')
        filedir2=[filedir2 '/'];
    end
    filedir = strcat(filedir2,filestruct(1).name);
    if(~isdir(filedir))
        files=[];
        filedates=[];
        return;
    end
    filestruct = dir(filedir);
end

if(isempty(filestruct))
    files = [];
    filedates=[];
    return;
end

filenames = {filestruct.name};
filedates = {filestruct.datenum};

notdots=~cellfun(@(x)(all(x=='.')),filenames);
filenames = filenames(notdots);
filedates = filedates(notdots);

if(case_sensitive)
    fmatch=~cellfun(@isempty,regexpi(filenames,fpattern,'matchcase'));
else
    fmatch=~cellfun(@isempty,regexpi(filenames,fpattern));
end
filenames = filenames(fmatch);
filedates = filedates(fmatch);

if(isempty(filenames))
    files = [];
    filedates=[];
    return;
end

if(filedir(end)~='/')
    filedir=[filedir '/'];
end
if(~isempty(removeprefix) && strcmp(removeprefix,filedir))
    files_tmp=filenames;
else
    files_tmp = strcat(filedir,filenames);
end
files_tmp = files_tmp(:);

files=files_tmp(:);
filedates=filedates(:);
