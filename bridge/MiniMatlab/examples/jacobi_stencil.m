% set width and height
width = 4000;
height = 4000;

%tic

% initialize grid
grid = zeros(width + 2, height + 2);
% set boundaries (freezetap)
grid(:,1) = -273.15;
grid(:,end) = -273.15;
grid(end,:) = -273.15;
grid(1,:) = 40;

for i=1:10,
    grid(2:end-1, 2:end-1) = 0.2 * (grid(2:end-1, 2:end-1) + grid(1:end-2, 2:end-1) + grid(3:end, 2:end-1) + grid(2:end-1, 3:end) + grid(2:end-1, 1:end-2));
end

%toc
%grid(:,2)
