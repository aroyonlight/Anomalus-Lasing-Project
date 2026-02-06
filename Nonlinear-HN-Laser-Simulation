clear; clc; close all;

%% ===================== GLOBAL PARAMETERS =====================
Lx = 6; Ly = 6;
N  = Lx * Ly;

t1     = 0.1;
gamma0 = 0.2;
Isat   = 1;

dt     = 1;
f0     = 0.01;

idx = @(x,y) (y-1)*Lx + x;

%% ===================== SYSTEM PARAMETERS =====================
delta_t = 0.04;
Gamma   = 1;

nSteps  = 50000;
nRelax  = 20000;
nstore  = 30000;

%% ===================== BUILD HATANO–NELSON H =====================
H = zeros(N,N);
for x = 1:Lx
    for y = 1:Ly
        i = idx(x,y);
        if x < Lx, H(i,idx(x+1,y)) = t1 - delta_t; end
        if x > 1,  H(i,idx(x-1,y)) = t1 + delta_t; end
        if y < Ly, H(i,idx(x,y+1)) = t1 - delta_t; end
        if y > 1,  H(i,idx(x,y-1)) = t1 + delta_t; end
    end
end

[V,D] = eig(H);
U0 = V * diag(exp(-1i*diag(D)*dt)) / V;

%% ===================== INITIAL CONDITION =====================
rng(123);
psi = f0 * (randn(N,1) + 1i*randn(N,1));

%% ===================== STORAGE =====================
fdata       = zeros(nstore,N);
Itime       = zeros(nSteps,1);
store_index = 0;

%% ===================== TIME EVOLUTION =====================
for iter = 1:nSteps

    % Gain + loss
    psi = psi .* exp(dt * (Gamma ./ (1 + abs(psi).^2) - gamma0));

    % Linear propagation
    psi = U0 * psi;

    % Total intensity
    Itime(iter) = sum(abs(psi).^2);

    % Store steady-state signals
    if iter > nRelax
        store_index = store_index + 1;
        fdata(store_index,:) = psi.';
    end
end

%% =======================================================================
%% 1) TIME EVOLUTION PLOT (INTENSITY vs TIME)
%% =======================================================================
time = (1:nSteps) * dt;

figure;
plot(time, Itime, 'LineWidth', 1.5);
xlabel('Time');
ylabel('Total intensity');
title('Time evolution of laser intensity');
grid on;

%% =======================================================================
%% 2) INTENSITY MAPPING (STEADY-STATE SPATIAL PROFILE)
%% =======================================================================
Iavg = mean(abs(fdata).^2,1);
Imap = reshape(Iavg, [Lx Ly]);

figure;
imagesc(Imap.');
axis equal tight;
set(gca,'YDir','normal');
colorbar;
xlabel('x');
ylabel('y');
title('Steady-state intensity distribution');

%% =======================================================================
%% 3) FFT LASING SPECTRUM
%% =======================================================================
powersp = zeros(nstore,1);
for j = 1:N
    sig = fdata(:,j);
    g   = fft(sig);
    powersp = powersp + abs(g).^2;
end

powersp = fftshift(powersp);
powersp = powersp / max(powersp);

freq = (-nstore/2:nstore/2-1).' * (2*pi/(dt*nstore));

figure;
semilogy(freq, powersp, 'LineWidth', 2);
xlabel('\omega');
ylabel('I_{out}(\omega)');
title(sprintf('Lasing spectrum (\\Gamma = %.2f)', Gamma));
xlim([-1 1]);
ylim([1e-6 1]);
grid on;

%% =======================================================================
%% 4) OUTPUT INTENSITY vs PUMP STRENGTH
%% =======================================================================
Gamma_range = linspace(0,1.0,30);
Iout = zeros(size(Gamma_range));

for gidx = 1:length(Gamma_range)

    Gamma = Gamma_range(gidx);
    psi = f0 * (randn(N,1) + 1i*randn(N,1));

    for step = 1:10000
        psi = psi .* exp(dt * (Gamma ./ (1 + abs(psi).^2) - gamma0));
        psi = U0 * psi;
    end

    Iout(gidx) = sum(abs(psi).^2);
end

figure;
plot(Gamma_range, Iout, '-o', 'LineWidth', 2);
xlabel('Pump strength \Gamma');
ylabel('Output intensity');
title('Laser input–output curve');
grid on;

%% =======================================================================
%% 5) LASER PHASE DIAGRAM 
%% =======================================================================
Gamma_range   = linspace(0,1.0,30);
delta_t_range = linspace(0,0.09999,20);

phase_map = zeros(length(delta_t_range), length(Gamma_range));
tol = 1e-3;

fprintf('\n=== Computing phase diagram ===\n');

for dt_idx = 1:length(delta_t_range)

    delta_t = delta_t_range(dt_idx);

    % Build H
    H = zeros(N,N);
    for x = 1:Lx
        for y = 1:Ly
            i = idx(x,y);
            if x < Lx, H(i,idx(x+1,y)) = t1 - delta_t; end
            if x > 1,  H(i,idx(x-1,y)) = t1 + delta_t; end
            if y < Ly, H(i,idx(x,y+1)) = t1 - delta_t; end
            if y > 1,  H(i,idx(x,y-1)) = t1 + delta_t; end
        end
    end

    [V,D] = eig(H);
    U0 = V * diag(exp(-1i*diag(D)*dt)) / V;

    for gidx = 1:length(Gamma_range)

        Gamma = Gamma_range(gidx);
        psi = f0 * (randn(N,1) + 1i*randn(N,1));

        for step = 1:5000
            psi = psi .* exp(dt * (Gamma ./ (1 + abs(psi).^2) - gamma0));
            psi = U0 * psi;
        end

        G_eff = Gamma ./ (1 + abs(psi).^2) - gamma0;
        H_eff = H + 1i * diag(G_eff);

        E = eig(H_eff);
        lasing_modes = find(imag(E) > -tol);

        if isempty(lasing_modes)
            phase_map(dt_idx,gidx) = 0;   % Subthreshold
        elseif length(lasing_modes) > 1
            phase_map(dt_idx,gidx) = 1;   % Multimode
        else
            phase_map(dt_idx,gidx) = 2;   % SSEM
        end
    end

    fprintf('Completed δt = %.3f\n', delta_t);
end

figure;
imagesc(Gamma_range, delta_t_range, phase_map);
set(gca,'YDir','normal');
colormap([
    0.1 0.1 0.5;
    0.3 0.6 0.9;
    1.0 1.0 0.4
]);
colorbar('Ticks',[0.5 1.5 2.5], ...
         'TickLabels',{'Subthreshold','Multimode','SSEM'});
xlabel('Pump strength \Gamma');
ylabel('\delta_t');
title('Laser phase diagram');
box on;
