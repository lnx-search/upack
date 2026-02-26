use super::generate::GeneratedSamples;

/// A routine that can be benchmarked and measured.
pub trait Routine: Default {
    type PreparedInput;

    /// The name/identifier for the routine.
    fn name() -> String;

    /// Prepare the input samples read for the timing run.
    fn prep(&mut self, samples: GeneratedSamples) -> Self::PreparedInput;

    /// Execute the routine for measurements.
    fn execute(&mut self, samples: &mut Self::PreparedInput) -> usize;
}
