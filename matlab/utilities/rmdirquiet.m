function rmdirquiet(x)

% function rmdirquiet(x)
%
% <x> refers to a file or directory location
%
% Delete <x>, suppressing warnings.
% If <x> doesn't exist, we just quietly return.
%
% example:
% mkdirquiet('test');
% mkdirquiet('test');
% rmdirquiet('test');
% rmdirquiet('test');

if exist(x,'dir')
  assert(rmdir(x,'s'),sprintf('rmdir of %s failed',x));
elseif exist(x,'file')
  delete(x);
end
