package golapack

import "github.com/whipstein/golinalg/mat"

// Dlaneg computes the Sturm count, the number of negative pivots
// encountered while factoring tridiagonal T - sigma I = L D L^T.
// This implementation works directly on the factors without forming
// the tridiagonal matrix T.  The Sturm count is also the number of
// eigenvalues of T less than sigma.
//
// This routine is called from DLARRB.
//
// The current routine does not use the PIVMIN parameter but rather
// requires IEEE-754 propagation of Infinities and NaNs.  This
// routine also has no input range restrictions but does require
// default exception handling such that x/0 produces Inf when x is
// non-zero, and Inf/Inf produces NaN.  For more information, see:
//
//   Marques, Riedy, and Voemel, "Benefits of IEEE-754 Features in
//   Modern Symmetric Tridiagonal Eigensolvers," SIAM Journal on
//   Scientific Computing, v28, n5, 2006.  DOI 10.1137/050641624
//   (Tech report version in LAWN 172 with the same title.)
func Dlaneg(n int, d, lld *mat.Vector, sigma, pivmin float64, r int) (dlanegReturn int) {
	var sawnan bool
	var bsav, dminus, dplus, gamma, one, p, t, tmp, zero float64
	var bj, blklen, j, neg1, neg2, negcnt int

	zero = 0.0
	one = 1.0
	//     Some architectures propagate Infinities and NaNs very slowly, so
	//     the code computes counts in BLKLEN chunks.  Then a NaN can
	//     propagate at most BLKLEN columns before being detected.  This is
	//     not a general tuning parameter; it needs only to be just large
	//     enough that the overhead is tiny in common cases.
	blklen = 128

	negcnt = 0
	//     I) upper part: L D L^T - SIGMA I = L+ D+ L+^T
	t = -sigma
	for bj = 1; bj <= r-1; bj += blklen {
		neg1 = 0
		bsav = t
		for j = bj; j <= min(bj+blklen-1, r-1); j++ {
			dplus = d.Get(j-1) + t
			if dplus < zero {
				neg1 = neg1 + 1
			}
			tmp = t / dplus
			t = tmp*lld.Get(j-1) - sigma
		}
		sawnan = Disnan(int(t))
		//     Run a slower version of the above loop if a NaN is detected.
		//     A NaN should occur only with a zero pivot after an infinite
		//     pivot.  In that case, substituting 1 for T/DPLUS is the
		//     correct limit.
		if sawnan {
			neg1 = 0
			t = bsav
			for j = bj; j <= min(bj+blklen-1, r-1); j++ {
				dplus = d.Get(j-1) + t
				if dplus < zero {
					neg1 = neg1 + 1
				}
				tmp = t / dplus
				if Disnan(int(tmp)) {
					tmp = one
				}
				t = tmp*lld.Get(j-1) - sigma
			}
		}
		negcnt = negcnt + neg1
	}

	//     II) lower part: L D L^T - SIGMA I = U- D- U-^T
	p = d.Get(n-1) - sigma
	for bj = n - 1; bj >= r; bj -= blklen {
		neg2 = 0
		bsav = p
		for j = bj; j >= max(bj-blklen+1, r); j-- {
			dminus = lld.Get(j-1) + p
			if dminus < zero {
				neg2 = neg2 + 1
			}
			tmp = p / dminus
			p = tmp*d.Get(j-1) - sigma
		}
		sawnan = Disnan(int(p))
		//     As above, run a slower version that substitutes 1 for Inf/Inf.

		if sawnan {
			neg2 = 0
			p = bsav
			for j = bj; j >= max(bj-blklen+1, r); j-- {
				dminus = lld.Get(j-1) + p
				if dminus < zero {
					neg2 = neg2 + 1
				}
				tmp = p / dminus
				if Disnan(int(tmp)) {
					tmp = one
				}
				p = tmp*d.Get(j-1) - sigma
			}
		}
		negcnt = negcnt + neg2
	}

	//     III) Twist index
	//       T was shifted by SIGMA initially.
	gamma = (t + sigma) + p
	if gamma < zero {
		negcnt = negcnt + 1
	}
	dlanegReturn = negcnt
	return
}
