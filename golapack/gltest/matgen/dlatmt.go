package matgen

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlatmt generates random matrices with specified singular values
//    (or symmetric/hermitian with specified eigenvalues)
//    for testing LAPACK programs.
//
//    DLATMT operates by applying the following sequence of
//    operations:
//
//      Set the diagonal to D, where D may be input or
//         computed according to MODE, COND, DMAX, and SYM
//         as described below.
//
//      Generate a matrix with the appropriate band structure, by one
//         of two methods:
//
//      Method A:
//          Generate a dense M x N matrix by multiplying D on the left
//              and the right by random unitary matrices, then:
//
//          Reduce the bandwidth according to KL and KU, using
//          Householder transformations.
//
//      Method B:
//          Convert the bandwidth-0 (i.e., diagonal) matrix to a
//              bandwidth-1 matrix using Givens rotations, "chasing"
//              out-of-band elements back, much as in QR; then
//              convert the bandwidth-1 to a bandwidth-2 matrix, etc.
//              Note that for reasonably small bandwidths (relative to
//              M and N) this requires less storage, as a dense matrix
//              is not generated.  Also, for symmetric matrices, only
//              one triangle is generated.
//
//      Method A is chosen if the bandwidth is a large fraction of the
//          order of the matrix, and LDA is at least M (so a dense
//          matrix can be stored.)  Method B is chosen if the bandwidth
//          is small (< 1/2 N for symmetric, < .3 N+M for
//          non-symmetric), or LDA is less than M and not less than the
//          bandwidth.
//
//      Pack the matrix if desired. Options specified by PACK are:
//         no packing
//         zero out upper half (if symmetric)
//         zero out lower half (if symmetric)
//         store the upper half columnwise (if symmetric or upper
//               triangular)
//         store the lower half columnwise (if symmetric or lower
//               triangular)
//         store the lower triangle in banded format (if symmetric
//               or lower triangular)
//         store the upper triangle in banded format (if symmetric
//               or upper triangular)
//         store the entire matrix in banded format
//      If Method B is chosen, and band format is specified, then the
//         matrix will be generated in the band format, so no repacking
//         will be necessary.
func Dlatmt(m, n *int, dist byte, iseed *[]int, sym byte, d *mat.Vector, mode *int, cond, dmax *float64, rank, kl, ku *int, pack byte, a *mat.Matrix, lda *int, work *mat.Vector, info *int) {
	var givens, ilextr, iltemp, topdwn bool
	var alpha, angle, c, dummy, extra, one, s, temp, twopi, zero float64
	var i, ic, icol, idist, iendch, iinfo, il, ilda, ioffg, ioffst, ipack, ipackg, ir, ir1, ir2, irow, irsign, iskew, isym, isympk, j, jc, jch, jkl, jku, jr, k, llb, minlda, mnmin, mr, nc, uub int

	zero = 0.0
	one = 1.0
	twopi = 2 * math.Pi

	//     1)      Decode and Test the input parameters.
	//             Initialize flags & seed.
	(*info) = 0

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 {
		return
	}

	//     Decode DIST
	if dist == 'U' {
		idist = 1
	} else if dist == 'S' {
		idist = 2
	} else if dist == 'N' {
		idist = 3
	} else {
		idist = -1
	}

	//     Decode SYM
	if sym == 'N' {
		isym = 1
		irsign = 0
	} else if sym == 'P' {
		isym = 2
		irsign = 0
	} else if sym == 'S' {
		isym = 2
		irsign = 1
	} else if sym == 'H' {
		isym = 2
		irsign = 1
	} else {
		isym = -1
	}

	//     Decode PACK
	isympk = 0
	if pack == 'N' {
		ipack = 0
	} else if pack == 'U' {
		ipack = 1
		isympk = 1
	} else if pack == 'L' {
		ipack = 2
		isympk = 1
	} else if pack == 'C' {
		ipack = 3
		isympk = 2
	} else if pack == 'R' {
		ipack = 4
		isympk = 3
	} else if pack == 'B' {
		ipack = 5
		isympk = 3
	} else if pack == 'Q' {
		ipack = 6
		isympk = 2
	} else if pack == 'Z' {
		ipack = 7
	} else {
		ipack = -1
	}

	//     Set certain internal parameters
	mnmin = minint(*m, *n)
	llb = minint(*kl, (*m)-1)
	uub = minint(*ku, (*n)-1)
	mr = minint(*m, (*n)+llb)
	nc = minint(*n, (*m)+uub)

	if ipack == 5 || ipack == 6 {
		minlda = uub + 1
	} else if ipack == 7 {
		minlda = llb + uub + 1
	} else {
		minlda = (*m)
	}

	//     Use Givens rotation method if bandwidth small enough,
	//     or if LDA is too small to store the matrix unpacked.
	givens = false
	if isym == 1 {
		if float64(llb+uub) < 0.3*float64(maxint(1, mr+nc)) {
			givens = true
		}
	} else {
		if 2*llb < (*m) {
			givens = true
		}
	}
	if (*lda) < (*m) && (*lda) >= minlda {
		givens = true
	}

	//     Set INFO if an error
	if (*m) < 0 {
		(*info) = -1
	} else if (*m) != (*n) && isym != 1 {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if idist == -1 {
		(*info) = -3
	} else if isym == -1 {
		(*info) = -5
	} else if absint(*mode) > 6 {
		(*info) = -7
	} else if ((*mode) != 0 && absint(*mode) != 6) && (*cond) < one {
		(*info) = -8
	} else if (*kl) < 0 {
		(*info) = -10
	} else if (*ku) < 0 || (isym != 1 && (*kl) != (*ku)) {
		(*info) = -11
	} else if ipack == -1 || (isympk == 1 && isym == 1) || (isympk == 2 && isym == 1 && (*kl) > 0) || (isympk == 3 && isym == 1 && (*ku) > 0) || (isympk != 0 && (*m) != (*n)) {
		(*info) = -12
	} else if (*lda) < maxint(1, minlda) {
		(*info) = -14
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DLATMT"), -(*info))
		return
	}

	//     Initialize random number generator
	for i = 1; i <= 4; i++ {
		(*iseed)[i-1] = absint((*iseed)[i-1]) % 4096
	}

	if (*iseed)[3]%2 != 1 {
		(*iseed)[3] = (*iseed)[3] + 1
	}

	//     2)      Set up D  if indicated.
	//
	//             Compute D according to COND and MODE
	Dlatm7(mode, cond, &irsign, &idist, iseed, d, &mnmin, rank, &iinfo)
	if iinfo != 0 {
		(*info) = 1
		return
	}

	//     Choose Top-Down if D is (apparently) increasing,
	//     Bottom-Up if D is (apparently) decreasing.
	if math.Abs(d.Get(0)) <= math.Abs(d.Get((*rank)-1)) {
		topdwn = true
	} else {
		topdwn = false
	}

	if (*mode) != 0 && absint(*mode) != 6 {
		//        Scale by DMAX
		temp = math.Abs(d.Get(0))
		for i = 2; i <= (*rank); i++ {
			temp = maxf64(temp, math.Abs(d.Get(i-1)))
		}

		if temp > zero {
			alpha = (*dmax) / temp
		} else {
			(*info) = 2
			return
		}

		goblas.Dscal(*rank, alpha, d, 1)
	}

	//     3)      Generate Banded Matrix using Givens rotations.
	//             Also the special case of UUB=LLB=0
	//
	//               Compute Addressing constants to cover all
	//               storage formats.  Whether GE, SY, GB, or SB,
	//               upper or lower triangle or both,
	//               the (i,j)-th element is in
	//               A( i - ISKEW*j + IOFFST, j )
	if ipack > 4 {
		ilda = (*lda) - 1
		iskew = 1
		if ipack > 5 {
			ioffst = uub + 1
		} else {
			ioffst = 1
		}
	} else {
		ilda = (*lda)
		iskew = 0
		ioffst = 0
	}

	//     IPACKG is the format that the matrix is generated in. If this is
	//     different from IPACK, then the matrix must be repacked at the
	//     end.  It also signals how to compute the norm, for scaling.
	ipackg = 0
	golapack.Dlaset('F', lda, n, &zero, &zero, a, lda)

	//     Diagonal Matrix -- We are done, unless it
	//     is to be stored SP/PP/TP (PACK='R' or 'C')
	if llb == 0 && uub == 0 {
		goblas.Dcopy(mnmin, d, 1, a.Vector(1-iskew+ioffst-1, 0), ilda+1)
		if ipack <= 2 || ipack >= 5 {
			ipackg = ipack
		}

	} else if givens {
		//        Check whether to use Givens rotations,
		//        Householder transformations, or nothing.
		if isym == 1 {
			//           Non-symmetric -- A = U D V
			if ipack > 4 {
				ipackg = ipack
			} else {
				ipackg = 0
			}

			goblas.Dcopy(mnmin, d, 1, a.Vector(1-iskew+ioffst-1, 0), ilda+1)

			if topdwn {
				jkl = 0
				for jku = 1; jku <= uub; jku++ {
					//                 Transform from bandwidth JKL, JKU-1 to JKL, JKU
					//
					//                 Last row actually rotated is M
					//                 Last column actually rotated is MIN( M+JKU, N )
					for jr = 1; jr <= minint((*m)+jku, *n)+jkl-1; jr++ {
						extra = zero
						angle = twopi * Dlarnd(func() *int { y := 1; return &y }(), iseed)
						c = math.Cos(angle)
						s = math.Sin(angle)
						icol = maxint(1, jr-jkl)
						if jr < (*m) {
							il = minint(*n, jr+jku) + 1 - icol
							Dlarot(true, jr > jkl, false, &il, &c, &s, a.Vector(jr-iskew*icol+ioffst-1, icol-1), &ilda, &extra, &dummy)
						}

						//                    Chase "EXTRA" back up
						ir = jr
						ic = icol
						for jch = jr - jkl; jch >= 1; jch -= (jkl + jku) {
							if ir < (*m) {
								golapack.Dlartg(a.GetPtr(ir+1-iskew*(ic+1)+ioffst-1, ic+1-1), &extra, &c, &s, &dummy)
							}
							irow = maxint(1, jch-jku)
							il = ir + 2 - irow
							temp = zero
							iltemp = jch > jku
							Dlarot(false, iltemp, true, &il, &c, func() *float64 { y := -s; return &y }(), a.Vector(irow-iskew*ic+ioffst-1, ic-1), &ilda, &temp, &extra)
							if iltemp {
								golapack.Dlartg(a.GetPtr(irow+1-iskew*(ic+1)+ioffst-1, ic+1-1), &temp, &c, &s, &dummy)
								icol = maxint(1, jch-jku-jkl)
								il = ic + 2 - icol
								extra = zero
								Dlarot(true, jch > jku+jkl, true, &il, &c, func() *float64 { y := -s; return &y }(), a.Vector(irow-iskew*icol+ioffst-1, icol-1), &ilda, &extra, &temp)
								ic = icol
								ir = irow
							}
						}
					}
				}

				jku = uub
				for jkl = 1; jkl <= llb; jkl++ {
					//                 Transform from bandwidth JKL-1, JKU to JKL, JKU
					for jc = 1; jc <= minint((*n)+jkl, *m)+jku-1; jc++ {
						extra = zero
						angle = twopi * Dlarnd(func() *int { y := 1; return &y }(), iseed)
						c = math.Cos(angle)
						s = math.Sin(angle)
						irow = maxint(1, jc-jku)
						if jc < (*n) {
							il = minint(*m, jc+jkl) + 1 - irow
							Dlarot(false, jc > jku, false, &il, &c, &s, a.Vector(irow-iskew*jc+ioffst-1, jc-1), &ilda, &extra, &dummy)
						}

						//                    Chase "EXTRA" back up
						ic = jc
						ir = irow
						for jch = jc - jku; jch >= 1; jch -= (jkl + jku) {
							if ic < (*n) {
								golapack.Dlartg(a.GetPtr(ir+1-iskew*(ic+1)+ioffst-1, ic+1-1), &extra, &c, &s, &dummy)
							}
							icol = maxint(1, jch-jkl)
							il = ic + 2 - icol
							temp = zero
							iltemp = jch > jkl
							Dlarot(true, iltemp, true, &il, &c, func() *float64 { y := -s; return &y }(), a.Vector(ir-iskew*icol+ioffst-1, icol-1), &ilda, &temp, &extra)
							if iltemp {
								golapack.Dlartg(a.GetPtr(ir+1-iskew*(icol+1)+ioffst-1, icol+1-1), &temp, &c, &s, &dummy)
								irow = maxint(1, jch-jkl-jku)
								il = ir + 2 - irow
								extra = zero
								Dlarot(false, jch > jkl+jku, true, &il, &c, func() *float64 { y := -s; return &y }(), a.Vector(irow-iskew*icol+ioffst-1, icol-1), &ilda, &extra, &temp)
								ic = icol
								ir = irow
							}
						}
					}
				}

			} else {
				//              Bottom-Up -- Start at the bottom right.
				jkl = 0
				for jku = 1; jku <= uub; jku++ {
					//                 Transform from bandwidth JKL, JKU-1 to JKL, JKU
					//
					//                 First row actually rotated is M
					//                 First column actually rotated is MIN( M+JKU, N )
					iendch = minint(*m, (*n)+jkl) - 1
					for jc = minint((*m)+jku, *n) - 1; jc >= 1-jkl; jc-- {
						extra = zero
						angle = twopi * Dlarnd(func() *int { y := 1; return &y }(), iseed)
						c = math.Cos(angle)
						s = math.Sin(angle)
						irow = maxint(1, jc-jku+1)
						if jc > 0 {
							il = minint(*m, jc+jkl+1) + 1 - irow
							Dlarot(false, false, jc+jkl < (*m), &il, &c, &s, a.Vector(irow-iskew*jc+ioffst-1, jc-1), &ilda, &dummy, &extra)
						}

						//                    Chase "EXTRA" back down
						ic = jc
						for _, jch = range genIter(jc+jkl, iendch, jkl+jku) {
							ilextr = ic > 0
							if ilextr {
								golapack.Dlartg(a.GetPtr(jch-iskew*ic+ioffst-1, ic-1), &extra, &c, &s, &dummy)
							}
							ic = maxint(1, ic)
							icol = minint((*n)-1, jch+jku)
							iltemp = jch+jku < (*n)
							temp = zero
							Dlarot(true, ilextr, iltemp, toPtr(icol+2-ic), &c, &s, a.Vector(jch-iskew*ic+ioffst-1, ic-1), &ilda, &extra, &temp)
							if iltemp {
								golapack.Dlartg(a.GetPtr(jch-iskew*icol+ioffst-1, icol-1), &temp, &c, &s, &dummy)
								il = minint(iendch, jch+jkl+jku) + 2 - jch
								extra = zero
								Dlarot(false, true, jch+jkl+jku <= iendch, &il, &c, &s, a.Vector(jch-iskew*icol+ioffst-1, icol-1), &ilda, &temp, &extra)
								ic = icol
							}
						}
					}
				}

				jku = uub
				for jkl = 1; jkl <= llb; jkl++ {
					//                 Transform from bandwidth JKL-1, JKU to JKL, JKU
					//
					//                 First row actually rotated is MIN( N+JKL, M )
					//                 First column actually rotated is N
					iendch = minint(*n, (*m)+jku) - 1
					for jr = minint((*n)+jkl, *m) - 1; jr >= 1-jku; jr-- {
						extra = zero
						angle = twopi * Dlarnd(func() *int { y := 1; return &y }(), iseed)
						c = math.Cos(angle)
						s = math.Sin(angle)
						icol = maxint(1, jr-jkl+1)
						if jr > 0 {
							il = minint(*n, jr+jku+1) + 1 - icol
							Dlarot(true, false, jr+jku < (*n), &il, &c, &s, a.Vector(jr-iskew*icol+ioffst-1, icol-1), &ilda, &dummy, &extra)
						}

						//                    Chase "EXTRA" back down
						ir = jr
						for _, jch = range genIter(jr+jku, iendch, jkl+jku) {
							ilextr = ir > 0
							if ilextr {
								golapack.Dlartg(a.GetPtr(ir-iskew*jch+ioffst-1, jch-1), &extra, &c, &s, &dummy)
							}
							ir = maxint(1, ir)
							irow = minint((*m)-1, jch+jkl)
							iltemp = jch+jkl < (*m)
							temp = zero
							Dlarot(false, ilextr, iltemp, toPtr(irow+2-ir), &c, &s, a.Vector(ir-iskew*jch+ioffst-1, jch-1), &ilda, &extra, &temp)
							if iltemp {
								golapack.Dlartg(a.GetPtr(irow-iskew*jch+ioffst-1, jch-1), &temp, &c, &s, &dummy)
								il = minint(iendch, jch+jkl+jku) + 2 - jch
								extra = zero
								Dlarot(true, true, jch+jkl+jku <= iendch, &il, &c, &s, a.Vector(irow-iskew*jch+ioffst-1, jch-1), &ilda, &temp, &extra)
								ir = irow
							}
						}
					}
				}
			}

		} else {
			//           Symmetric -- A = U D U'
			// ipackg = ipack
			ioffg = ioffst

			if topdwn {
				//              Top-Down -- Generate Upper triangle only
				if ipack >= 5 {
					ipackg = 6
					ioffg = uub + 1
				} else {
					ipackg = 1
				}
				goblas.Dcopy(mnmin, d, 1, a.Vector(1-iskew+ioffg-1, 0), ilda+1)

				for k = 1; k <= uub; k++ {
					for jc = 1; jc <= (*n)-1; jc++ {
						irow = maxint(1, jc-k)
						il = minint(jc+1, k+2)
						extra = zero
						temp = a.Get(jc-iskew*(jc+1)+ioffg-1, jc+1-1)
						angle = twopi * Dlarnd(func() *int { y := 1; return &y }(), iseed)
						c = math.Cos(angle)
						s = math.Sin(angle)
						Dlarot(false, jc > k, true, &il, &c, &s, a.Vector(irow-iskew*jc+ioffg-1, jc-1), &ilda, &extra, &temp)
						Dlarot(true, true, false, toPtr(minint(k, (*n)-jc)+1), &c, &s, a.Vector((1-iskew)*jc+ioffg-1, jc-1), &ilda, &temp, &dummy)

						//                    Chase EXTRA back up the matrix
						icol = jc
						for jch = jc - k; jch >= 1; jch -= k {
							golapack.Dlartg(a.GetPtr(jch+1-iskew*(icol+1)+ioffg-1, icol+1-1), &extra, &c, &s, &dummy)
							temp = a.Get(jch-iskew*(jch+1)+ioffg-1, jch+1-1)
							Dlarot(true, true, true, toPtr(k+2), &c, func() *float64 { y := -s; return &y }(), a.Vector((1-iskew)*jch+ioffg-1, jch-1), &ilda, &temp, &extra)
							irow = maxint(1, jch-k)
							il = minint(jch+1, k+2)
							extra = zero
							Dlarot(false, jch > k, true, &il, &c, func() *float64 { y := -s; return &y }(), a.Vector(irow-iskew*jch+ioffg-1, jch-1), &ilda, &extra, &temp)
							icol = jch
						}
					}
				}

				//              If we need lower triangle, copy from upper. Note that
				//              the order of copying is chosen to work for 'q' -> 'b'
				if ipack != ipackg && ipack != 3 {
					for jc = 1; jc <= (*n); jc++ {
						irow = ioffst - iskew*jc
						for jr = jc; jr <= minint(*n, jc+uub); jr++ {
							a.Set(jr+irow-1, jc-1, a.Get(jc-iskew*jr+ioffg-1, jr-1))
						}
					}
					if ipack == 5 {
						for jc = (*n) - uub + 1; jc <= (*n); jc++ {
							for jr = (*n) + 2 - jc; jr <= uub+1; jr++ {
								a.Set(jr-1, jc-1, zero)
							}
						}
					}
					if ipackg == 6 {
						ipackg = ipack
					} else {
						ipackg = 0
					}
				}
			} else {
				//              Bottom-Up -- Generate Lower triangle only
				if ipack >= 5 {
					ipackg = 5
					if ipack == 6 {
						ioffg = 1
					}
				} else {
					ipackg = 2
				}
				goblas.Dcopy(mnmin, d, 1, a.Vector(1-iskew+ioffg-1, 0), ilda+1)

				for k = 1; k <= uub; k++ {
					for jc = (*n) - 1; jc >= 1; jc-- {
						il = minint((*n)+1-jc, k+2)
						extra = zero
						temp = a.Get(1+(1-iskew)*jc+ioffg-1, jc-1)
						angle = twopi * Dlarnd(func() *int { y := 1; return &y }(), iseed)
						c = math.Cos(angle)
						s = -math.Sin(angle)
						Dlarot(false, true, (*n)-jc > k, &il, &c, &s, a.Vector((1-iskew)*jc+ioffg-1, jc-1), &ilda, &temp, &extra)
						icol = maxint(1, jc-k+1)
						Dlarot(true, false, true, toPtr(jc+2-icol), &c, &s, a.Vector(jc-iskew*icol+ioffg-1, icol-1), &ilda, &dummy, &temp)

						//                    Chase EXTRA back down the matrix
						icol = jc
						for _, jch = range genIter(jc+k, (*n)-1, k) {
							golapack.Dlartg(a.GetPtr(jch-iskew*icol+ioffg-1, icol-1), &extra, &c, &s, &dummy)
							temp = a.Get(1+(1-iskew)*jch+ioffg-1, jch-1)
							Dlarot(true, true, true, toPtr(k+2), &c, &s, a.Vector(jch-iskew*icol+ioffg-1, icol-1), &ilda, &extra, &temp)
							il = minint((*n)+1-jch, k+2)
							extra = zero
							Dlarot(false, true, (*n)-jch > k, &il, &c, &s, a.Vector((1-iskew)*jch+ioffg-1, jch-1), &ilda, &temp, &extra)
							icol = jch
						}
					}
				}

				//              If we need upper triangle, copy from lower. Note that
				//              the order of copying is chosen to work for 'b' -> 'q'
				if ipack != ipackg && ipack != 4 {
					for jc = (*n); jc >= 1; jc-- {
						irow = ioffst - iskew*jc
						for jr = jc; jr >= maxint(1, jc-uub); jr-- {
							a.Set(jr+irow-1, jc-1, a.Get(jc-iskew*jr+ioffg-1, jr-1))
						}
					}
					if ipack == 6 {
						for jc = 1; jc <= uub; jc++ {
							for jr = 1; jr <= uub+1-jc; jr++ {
								a.Set(jr-1, jc-1, zero)
							}
						}
					}
					if ipackg == 5 {
						ipackg = ipack
					} else {
						ipackg = 0
					}
				}
			}
		}

	} else {
		//        4)      Generate Banded Matrix by first
		//                Rotating by random Unitary matrices,
		//                then reducing the bandwidth using Householder
		//                transformations.
		//
		//                Note: we should get here only if LDA .ge. N
		if isym == 1 {
			//           Non-symmetric -- A = U D V
			Dlagge(&mr, &nc, &llb, &uub, d, a, lda, iseed, work, &iinfo)
		} else {
			//           Symmetric -- A = U D U'
			Dlagsy(m, &llb, d, a, lda, iseed, work, &iinfo)

		}
		if iinfo != 0 {
			(*info) = 3
			return
		}
	}

	//     5)      Pack the matrix
	if ipack != ipackg {
		if ipack == 1 {
			//           'U' -- Upper triangular, not packed
			for j = 1; j <= (*m); j++ {
				for i = j + 1; i <= (*m); i++ {
					a.Set(i-1, j-1, zero)
				}
			}

		} else if ipack == 2 {
			//           'L' -- Lower triangular, not packed
			for j = 2; j <= (*m); j++ {
				for i = 1; i <= j-1; i++ {
					a.Set(i-1, j-1, zero)
				}
			}

		} else if ipack == 3 {
			//           'C' -- Upper triangle packed Columnwise.
			icol = 1
			irow = 0
			for j = 1; j <= (*m); j++ {
				for i = 1; i <= j; i++ {
					irow = irow + 1
					if irow > (*lda) {
						irow = 1
						icol = icol + 1
					}
					a.Set(irow-1, icol-1, a.Get(i-1, j-1))
				}
			}

		} else if ipack == 4 {
			//           'R' -- Lower triangle packed Columnwise.
			icol = 1
			irow = 0
			for j = 1; j <= (*m); j++ {
				for i = j; i <= (*m); i++ {
					irow = irow + 1
					if irow > (*lda) {
						irow = 1
						icol = icol + 1
					}
					a.Set(irow-1, icol-1, a.Get(i-1, j-1))
				}
			}

		} else if ipack >= 5 {
			//           'B' -- The lower triangle is packed as a band matrix.
			//           'Q' -- The upper triangle is packed as a band matrix.
			//           'Z' -- The whole matrix is packed as a band matrix.
			if ipack == 5 {
				uub = 0
			}
			if ipack == 6 {
				llb = 0
			}

			for j = 1; j <= uub; j++ {
				for i = minint(j+llb, *m); i >= 1; i-- {
					a.Set(i-j+uub+1-1, j-1, a.Get(i-1, j-1))
				}
			}

			for j = uub + 2; j <= (*n); j++ {
				for i = j - uub; i <= minint(j+llb, *m); i++ {
					a.Set(i-j+uub+1-1, j-1, a.Get(i-1, j-1))
				}
			}
		}

		//        If packed, zero out extraneous elements.
		//
		//        Symmetric/Triangular Packed --
		//        zero out everything after A(IROW,ICOL)
		if ipack == 3 || ipack == 4 {
			for jc = icol; jc <= (*m); jc++ {
				for jr = irow + 1; jr <= (*lda); jr++ {
					a.Set(jr-1, jc-1, zero)
				}
				irow = 0
			}

		} else if ipack >= 5 {
			//           Packed Band --
			//              1st row is now in A( UUB+2-j, j), zero above it
			//              m-th row is now in A( M+UUB-j,j), zero below it
			//              last non-zero diagonal is now in A( UUB+LLB+1,j ),
			//                 zero below it, too.
			ir1 = uub + llb + 2
			ir2 = uub + (*m) + 2
			for jc = 1; jc <= (*n); jc++ {
				for jr = 1; jr <= uub+1-jc; jr++ {
					a.Set(jr-1, jc-1, zero)
				}
				for jr = maxint(1, minint(ir1, ir2-jc)); jr <= (*lda); jr++ {
					a.Set(jr-1, jc-1, zero)
				}
			}
		}
	}
}
