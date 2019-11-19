function [R_fit G_fit B_fit, OPDaxis_fit] = JLfitSplineRGB(R, G, B, eval_min, eval_max, n_samples, OPDaxis)

%%
if(0)
    figure
    plot3(R, G, B)
    xlabel 'R'
    ylabel 'G'
    zlabel 'B'
end

if(0)
    figure
    plot(R)
    hold on
    plot(G), plot(B)
    hold off
    legend('R', 'G', 'B')
    xlabel('OPD')
    ylabel('Intensity')
end


%% fit spline curve to data and watch the effect
xyz = cat(2, R', G', B');
% subsample the data collection and then try to fit it with a spline curve
xyzfitpt = xyz(1:2:end, :);
xyzfit = cscvn(xyzfitpt');
%t_range = linspace(min(min(xyz)), max(max(xyz)), numel(R));
%t_range = linspace(min(min(xyz)), max(max(xyz)), numel(R));
t_range = linspace(eval_min, eval_max, n_samples); %% TODO: Needs to be adjusted! 
OPDaxis_fit = linspace(min(OPDaxis), max(OPDaxis), n_samples);
xyzfiteval = fnval(xyzfit, t_range);

% plot result of fitted and orignial curves
figure, 
plot3(xyzfiteval(1, :), xyzfiteval(2, :), xyzfiteval(3, :), 'LineWidth',2), hold on
plot3(R, G, B, 'g', 'LineWidth',2), hold off
legend ({'RGB fit', 'RGB raw'},'FontSize',12,'TextColor','blue')
xlabel('Red - Pixelvalue','fontsize',18)
ylabel('Green - Pixelvalue','fontsize',18)
zlabel('Blue - Pixelvalue','fontsize',18)
grid on
set(gcf,'color','w');

R_fit = xyzfiteval(1, :);
G_fit = xyzfiteval(2, :);
B_fit = xyzfiteval(3, :);

