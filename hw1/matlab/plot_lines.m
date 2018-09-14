clear
rhoScale = 1:2:50;
thetaScale = 0:5:360;
thetaNum = length(thetaScale);

x = [10, 15, 30];
y = [10, 15, 30];
pointNum = length(x);

lines = zeros(thetaNum, pointNum);
for i = 1:pointNum
    for t = 1:thetaNum
        theta = thetaScale(t);
        lines(t, i) = x(i)*cosd(theta)+y(i)*sind(theta);
    end
end

figure(1)
plot(thetaScale', lines(:,1), thetaScale', lines(:,2), thetaScale', lines(:,3));