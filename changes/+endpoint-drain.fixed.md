When a concurrent endpoint run hits an error, items before the failing one that were still in flight are now waited for and written, as documented, so a resume does not request them again.
