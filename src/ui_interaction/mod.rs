//! 求解器控制和诊断状态。

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverCommand { Run, Pause, Reset, Step }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverStatus { Idle, Running, Paused, Converged, Failed }

#[derive(Clone, Debug)]
pub struct SolverControl {
    pub status: SolverStatus,
    pub command: Option<SolverCommand>,
    pub iteration: usize,
    pub max_iterations: usize,
}

impl Default for SolverControl {
    fn default() -> Self { Self { status: SolverStatus::Idle, command: None, iteration: 0, max_iterations: 1000 } }
}

impl SolverControl {
    pub fn submit(&mut self, command: SolverCommand) { self.command = Some(command); }
    pub fn take_command(&mut self) -> Option<SolverCommand> { self.command.take() }
}
