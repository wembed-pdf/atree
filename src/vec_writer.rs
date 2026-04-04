use std::mem::MaybeUninit;

/// Zero-overhead batched writer into a `Vec`'s spare capacity.
///
/// Wraps a `&mut Vec<T>` and tracks a write cursor beyond `len()`.
/// Capacity is managed externally via [`ensure_capacity`](Self::ensure_capacity),
/// and the inner loop uses [`next_chunk_unchecked`](Self::next_chunk_unchecked)
/// to hand out fixed-size windows with no branch.
///
/// The written elements are committed when [`finish`](Self::finish) is called,
/// or automatically on drop as a safety net.
pub(crate) struct VecWriter<'a, T> {
    vec: &'a mut Vec<T>,
    len: usize,
}

impl<'a, T> VecWriter<'a, T> {
    #[inline(always)]
    pub(crate) fn new(vec: &'a mut Vec<T>) -> Self {
        let len = vec.len();
        Self { vec, len }
    }

    /// Ensure there is room for at least `additional` more elements
    /// beyond the current cursor position.
    #[inline(always)]
    pub(crate) fn ensure_capacity(&mut self, additional: usize) {
        let required = self.len + additional;
        if required > self.vec.capacity() {
            self.vec.reserve(required - self.vec.len());
        }
    }

    /// Return the next `W`-element window as `&mut [MaybeUninit<T>; W]`.
    ///
    /// # Safety
    ///
    /// The caller must have previously called [`ensure_capacity`](Self::ensure_capacity)
    /// with enough room so that `self.len + W <= self.vec.capacity()`.
    #[inline(always)]
    pub(crate) unsafe fn next_chunk_unchecked<const W: usize>(&mut self) -> &mut [MaybeUninit<T>; W] {
        debug_assert!(
            self.len + W <= self.vec.capacity(),
            "VecWriter: cursor {} + chunk {} exceeds capacity {}",
            self.len,
            W,
            self.vec.capacity(),
        );
        let ptr = self.vec.as_mut_ptr().wrapping_add(self.len) as *mut MaybeUninit<T>;
        // SAFETY: ensure_capacity was called so ptr..ptr+W is within the allocation.
        // The returned slice is MaybeUninit so no initialization requirement.
        unsafe { &mut *(ptr as *mut [MaybeUninit<T>; W]) }
    }

    /// Advance the cursor by `n` elements (the number actually written
    /// into the last chunk).
    ///
    /// # Safety
    ///
    /// The caller must ensure that `n` elements starting at the current
    /// cursor position have been initialized (e.g. via [`next_chunk_unchecked`](Self::next_chunk_unchecked)).
    #[inline(always)]
    pub(crate) unsafe fn advance(&mut self, n: usize) {
        debug_assert!(
            self.len + n <= self.vec.capacity(),
            "VecWriter: advance would exceed capacity"
        );
        self.len += n;
    }

    /// Commit the final length to the underlying `Vec` and return it.
    #[inline(always)]
    pub(crate) fn finish(self) -> usize {
        let written = self.len;
        // SAFETY: all elements in vec[old_len..self.len] were initialized
        // by the caller via next_chunk_unchecked + MaybeUninit::write.
        unsafe { self.vec.set_len(written) };
        // Prevent drop from running set_len again.
        std::mem::forget(self);
        written
    }
}

impl<T> Drop for VecWriter<'_, T> {
    fn drop(&mut self) {
        // Safety net: if the caller didn't call finish() (e.g. early return
        // or panic), commit whatever was written so far rather than leaking.
        // SAFETY: same invariant — caller must only advance past initialized elements.
        unsafe { self.vec.set_len(self.len) };
    }
}
