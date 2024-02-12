{% raw %} 

$$
\newcommand{\diff}{\mathop{}\!\mathrm{d}}								% Differential
\newcommand{\pdiff}[2]{ { \frac{\partial {#1}}{\partial {#2}} } }		% Partial differentiation
\newcommand{\D}{\ensuremath{\mathcal{D}}}								% Generalized derivative
\newcommand{\eps}{{\varepsilon}}										% Epsilon
\newcommand{\ball}{{\mathcal B}}										% Ball
\newcommand{\clip}{{\text{clip}}}										% Clip function
\newcommand{\Lip}[1]{\text{Lip}_{#1}}									% Lipschitz constant of #1
\newcommand{\sgn}{{\text{sgn}}}											% Signum function
\newcommand{\diam}{{\text{diam}}}										% Diameter
\newcommand{\dom}{{\text{dom}}}											% Domain
\newcommand{\ramp}{{\text{ramp}}}										% Ramp	
\newcommand{\co}{{\overline{\text{co}}}}								% Convex closure
\DeclareMathOperator*{\argmin}{\text{arg\,min}}							% Argmin
\DeclareMathOperator*{\argmax}{\text{arg\,max}}							% Argmax
%\newcommand{\ln}{\text{ln}}												% Natural logarithm
\newcommand{\transp}{\ensuremath{^{\top}}}								% Matrix transpose
\newcommand{\inv}{\ensuremath{^{-1}}}									% Inverse
\newcommand{\tovec}[1]{\ensuremath{\text{vec}}\left(#1\right)}			% To-vector transformation
\newcommand{\nrm}[1]{\left\lVert#1\right\rVert}							% Norm
\newcommand{\diag}[1]{{\text{diag}}\left(#1\right)}						% Diagonal
\newcommand{\abs}[1]{\left\lvert#1\right\rvert}							% Absolute value
\newcommand{\scal}[1]{\left\langle#1\right\rangle}						% Scalar product
\newcommand{\tr}[1]{{\text{tr}}\left(#1\right)}							% Trace
\newcommand{\E}[2][{}]{\mathbb E_{#1}\left[#2\right]}					% Mean
\newcommand{\Es}[2][{}]{\hat {\mathbb E}_{#1}\left[#2\right]}			% Sample mean
\newcommand{\PP}[1]{\mathbb P\left[#1\right]}							% Probability
\newcommand{\bigo}[1]{\mathcal O\left(#1\right)}						% Big-o
\newcommand{\low}{{\text{low}}}											% Lower bound
\newcommand{\up}{{\text{up}}}											% Upper bound
\newcommand{\ra}{\rightarrow}											% Right arrow
\newcommand{\la}{\leftarrow}											% Left arrow
\newcommand{\rra}{\rightrightarrows}									% Double right arrow
\newcommand{\ie}{\unskip, i.\,e.,\xspace}								% That is
\newcommand{\eg}{\unskip, e.\,g.,\xspace}								% For example
\newcommand{\sut}{\text{s.\,t.\,}}										% Such that or subject to
\newcommand{\wrt}{w.\,r.\,t. \xspace}									% With respect to
\let\oldemptyset\emptyset
\let\emptyset\varnothing
\newcommand{\N}{{\mathbb{N}}}											% Set of natural numbers
\newcommand{\Z}{{\mathbb{Z}}}											% Set of integer numbers
\newcommand{\Q}{{\mathbb{Q}}}											% Set of rational numbers
\newcommand{\R}{{\mathbb{R}}}											% Set of real numbers
%\newcommand{\red}[1]{\textcolor{red}{#1}}
%\newcommand{\blue}[1]{\textcolor{blue}{#1}}
%\definecolor{dgreen}{rgb}{0.0, 0.5, 0.0}
%\newcommand{\green}[1]{\textcolor{dgreen}{#1}}
\newcommand{\state}{s}													% State (as vector)
\newcommand{\State}{S}													% State (as random variable)
\newcommand{\states}{\mathbb S}											% State space
\newcommand{\action}{a}													% Action (as vector)	
\newcommand{\Action}{A}													% Action (as random variable)
\newcommand{\actions}{\mathbb A}										% Action space
\newcommand{\traj}{z}													% State-action tuple (as vector tuple)
\newcommand{\Traj}{Z}													% State-action tuple (as random variable tuple)
\newcommand{\obs}{o}													% Observation (as vector)
\newcommand{\Obs}{O}													% Observation (as random variable)
\newcommand{\obses}{\mathbb O}											% Observation space
\newcommand{\policy}{\pi}												% Policy (as function or distribution)
\newcommand{\policies}{\Pi}												% Policy space
\newcommand{\transit}{p}												% State transition map
\newcommand{\reward}{r}													% Reward (as vector)
\newcommand{\Reward}{R}													% Reward (as random varaible)
\newcommand{\cost}{c}													% Cost (as vector)
\newcommand{\Cost}{C}													% Cost (as random varaible)
\newcommand{\Value}{V}													% Value
\newcommand{\Advan}{\mathcal A}											% Advantage
\newcommand{\W}{\ensuremath{\mathbb{W}}}								% Weight space
\newcommand{\B}{\ensuremath{\mathbb{B}}}								% Basin
\newcommand{\G}{\ensuremath{\mathbb{G}}}								% Attractor (goal set)
\newcommand{\Hamilt}{\ensuremath{\mathcal{H}}}							% Hamiltonian
\newcommand{\K}{\ensuremath{\mathcal{K}}\xspace}						% Class kappa
\newcommand{\KL}{\ensuremath{\mathcal{KL}}\xspace}						% Class kappa-ell
\newcommand{\Kinf}{\ensuremath{\mathcal{K}_{\infty}}\xspace}			% Class kappa-infinity
\newcommand{\KLinf}{\ensuremath{\mathcal{KL}_{\infty}}\xspace}			% Class kappa-ell-infinity
\newcommand{\T}{\mathcal T}												% Total time
\newcommand{\deltau}{\Delta \tau}										% Time step size
\newcommand{\dt}{\ensuremath{\mathrm{d}t}}								% Time differential
\newcommand{\normpdf}{\ensuremath{\mathcal N}}							% Normal PDF
\newcommand{\trajpdf}{\rho}												% State-action PDF
\newcommand{\TD}{\delta}												% Temporal difference
\newcommand{\old}{{\text{old}}}											% Old (previous) index
\newcommand{\loss}{\mathcal L}											% Loss
\newcommand{\replay}{\mathcal R}										% Experience replay
\newcommand{\safeset}{\ensuremath{\mathcal{S}}}							% Safe set
\newcommand{\dkappa}{\kappa_{\text{dec}}}								% Decay kappa function
\DeclarePairedDelimiterX{\infdivx}[2]{(}{)}{%
  #1\;\delimsize\|\;#2%
}
\newcommand{\kldiv}{d_{\text{KL}}\infdivx}								% KL-divergence
\newcommand{\barkldiv}{\bar d_{\text{KL}}\infdivx}						% Average KL-divergence
\newcommand{\spc}{{\,\,}}												% White space to be used in logical formulas
$$

{% endraw %} 