package matgen

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlatms generates random matrices with specified singular values
//    (or hermitian with specified eigenvalues)
//    for testing LAPACK programs.
//
//    ZLATMS operates by applying the following sequence of
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
//              Householder transformations.
//
//      Method B:
//          Convert the bandwidth-0 (i.e., diagonal) matrix to a
//              bandwidth-1 matrix using Givens rotations, "chasing"
//              out-of-band elements back, much as in QR; then convert
//              the bandwidth-1 to a bandwidth-2 matrix, etc.  Note
//              that for reasonably small bandwidths (relative to M and
//              N) this requires less storage, as a dense matrix is not
//              generated.  Also, for hermitian or symmetric matrices,
//              only one triangle is generated.
//
//      Method A is chosen if the bandwidth is a large fraction of the
//          order of the matrix, and LDA is at least M (so a dense
//          matrix can be stored.)  Method B is chosen if the bandwidth
//          is small (< 1/2 N for hermitian or symmetric, < .3 N+M for
//          non-symmetric), or LDA is less than M and not less than the
//          bandwidth.
//
//      Pack the matrix if desired. Options specified by PACK are:
//         no packing
//         zero out upper half (if hermitian)
//         zero out lower half (if hermitian)
//         store the upper half columnwise (if hermitian or upper
//               triangular)
//         store the lower half columnwise (if hermitian or lower
//               triangular)
//         store the lower triangle in banded format (if hermitian or
//               lower triangular)
//         store the upper triangle in banded format (if hermitian or
//               upper triangular)
//         store the entire matrix in banded format
//      If Method B is chosen, and band format is specified, then the
//         matrix will be generated in the band format, so no repacking
//         will be necessary.
func Zlatms(m, n int, dist byte, iseed *[]int, sym byte, d *mat.Vector, mode int, cond, dmax float64, kl, ku int, pack byte, a *mat.CMatrix, work *mat.CVector) (err error) {
	var givens, ilextr, iltemp, topdwn, zsym bool
	var c, ct, ctemp, czero, dummy, extra, s, st complex128
	var alpha, angle, one, realc, temp, twopi, zero float64
	var i, ic, icol, idist, iendch, iinfo, il, ilda, ioffg, ioffst, ipack, ipackg, ir, ir1, ir2, irow, irsign, iskew, isym, isympk, j, jc, jch, jkl, jku, jr, k, llb, minlda, mnmin, mr, nc, uub int

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	twopi = 6.2831853071795864769252867663

	//     1)      Decode and Test the input parameters.
	//             Initialize flags & seed.
	//     Quick return if possible
	if m == 0 || n == 0 {
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
		zsym = false
	} else if sym == 'P' {
		isym = 2
		irsign = 0
		zsym = false
	} else if sym == 'S' {
		isym = 2
		irsign = 0
		zsym = true
	} else if sym == 'H' {
		isym = 2
		irsign = 1
		zsym = false
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
	mnmin = min(m, n)
	llb = min(kl, m-1)
	uub = min(ku, n-1)
	mr = min(m, n+llb)
	nc = min(n, m+uub)

	if ipack == 5 || ipack == 6 {
		minlda = uub + 1
	} else if ipack == 7 {
		minlda = llb + uub + 1
	} else {
		minlda = m
	}

	//     Use Givens rotation method if bandwidth small enough,
	//     or if LDA is too small to store the matrix unpacked.
	givens = false
	if isym == 1 {
		if float64(llb+uub) < 0.3*float64(max(1, mr+nc)) {
			givens = true
		}
	} else {
		if 2*llb < m {
			givens = true
		}
	}
	if a.Rows < m && a.Rows >= minlda {
		givens = true
	}

	//     Set INFO if an error
	if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if m != n && isym != 1 {
		err = fmt.Errorf("m != n && isym != 1: sym='%c', m=%v, n=%v", sym, m, n)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if idist == -1 {
		err = fmt.Errorf("idist == -1: dist='%c'", dist)
	} else if isym == -1 {
		err = fmt.Errorf("isym == -1: sym='%c'", sym)
	} else if abs(mode) > 6 {
		err = fmt.Errorf("abs(mode) > 6: mode=%v", mode)
	} else if (mode != 0 && abs(mode) != 6) && cond < one {
		err = fmt.Errorf("(mode != 0 && abs(mode) != 6) && cond < one: mode=%v, cond=%v", mode, cond)
	} else if kl < 0 {
		err = fmt.Errorf("kl < 0: kl=%v", kl)
	} else if ku < 0 || (isym != 1 && kl != ku) {
		err = fmt.Errorf("ku < 0 || (isym != 1 && kl != ku): sym='%c', kl=%v, ku=%v", sym, kl, ku)
	} else if ipack == -1 || (isympk == 1 && isym == 1) || (isympk == 2 && isym == 1 && kl > 0) || (isympk == 3 && isym == 1 && ku > 0) || (isympk != 0 && m != n) {
		err = fmt.Errorf("ipack == -1 || (isympk == 1 && isym == 1) || (isympk == 2 && isym == 1 && kl > 0) || (isympk == 3 && isym == 1 && ku > 0) || (isympk != 0 && m != n): pack='%c', sym='%c', kl=%v, ku=%v, m=%v, n=%v", pack, sym, kl, ku, m, n)
	} else if a.Rows < max(1, minlda) {
		err = fmt.Errorf("a.Rows < max(1, minlda): a.Rows=%v, minlda=%v", a.Rows, minlda)
	}

	if err != nil {
		gltest.Xerbla2("Zlatms", err)
		return
	}

	//     Initialize random number generator
	for i = 1; i <= 4; i++ {
		(*iseed)[i-1] = (int(math.Abs(float64((*iseed)[i-1]))) % 4096)
	}

	if ((*iseed)[3] % 2) != 1 {
		(*iseed)[3] = (*iseed)[3] + 1
	}

	//     2)      Set up D  if indicated.
	//
	//             Compute D according to COND and MODE
	if err = Dlatm1(mode, cond, irsign, idist, iseed, d, mnmin); err != nil {
		err = fmt.Errorf("Error return from Dlatm1")
		return
	}

	//     Choose Top-Down if D is (apparently) increasing,
	//     Bottom-Up if D is (apparently) decreasing.
	if math.Abs(d.Get(0)) <= math.Abs(d.Get(mnmin-1)) {
		topdwn = true
	} else {
		topdwn = false
	}

	if mode != 0 && abs(mode) != 6 {
		//        Scale by DMAX
		temp = math.Abs(d.Get(0))
		for i = 2; i <= mnmin; i++ {
			temp = math.Max(temp, math.Abs(d.Get(i-1)))
		}

		if temp > zero {
			alpha = dmax / temp
		} else {
			err = fmt.Errorf("Cannot scale to dmax (max. sing. value is 0)")
			return
		}

		d.Scal(mnmin, alpha, 1)

	}

	golapack.Zlaset(Full, a.Rows, n, czero, czero, a)

	//     3)      Generate Banded Matrix using Givens rotations.
	//             Also the special case of UUB=LLB=0
	//
	//               Compute Addressing constants to cover all
	//               storage formats.  Whether GE, HE, SY, GB, HB, or SB,
	//               upper or lower triangle or both,
	//               the (i,j)-th element is in
	//               A( i - ISKEW*j + IOFFST, j )
	if ipack > 4 {
		ilda = a.Rows - 1
		iskew = 1
		if ipack > 5 {
			ioffst = uub + 1
		} else {
			ioffst = 1
		}
	} else {
		ilda = a.Rows
		iskew = 0
		ioffst = 0
	}

	//     IPACKG is the format that the matrix is generated in. If this is
	//     different from IPACK, then the matrix must be repacked at the
	//     end.  It also signals how to compute the norm, for scaling.
	ipackg = 0

	//     Diagonal Matrix -- We are done, unless it
	//     is to be stored HP/SP/PP/TP (PACK='R' or 'C')
	if llb == 0 && uub == 0 {
		for j = 1; j <= mnmin; j++ {
			a.Set((1-iskew)*j+ioffst-1, j-1, complex(d.Get(j-1), 0))
		}

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

			for j = 1; j <= mnmin; j++ {
				a.Set((1-iskew)*j+ioffst-1, j-1, complex(d.Get(j-1), 0))
			}

			if topdwn {
				jkl = 0
				for jku = 1; jku <= uub; jku++ {
					//                 Transform from bandwidth JKL, JKU-1 to JKL, JKU
					//
					//                 Last row actually rotated is M
					//                 Last column actually rotated is MIN( M+JKU, N )
					for jr = 1; jr <= min(m+jku, n)+jkl-1; jr++ {
						extra = czero
						angle = twopi * Dlarnd(1, iseed)
						c = complex(math.Cos(angle), 0) * Zlarnd(5, *iseed)
						s = complex(math.Sin(angle), 0) * Zlarnd(5, *iseed)
						icol = max(1, jr-jkl)
						if jr < m {
							il = min(n, jr+jku) + 1 - icol
							extra, dummy = Zlarot(true, jr > jkl, false, il, c, s, a.Off(jr-iskew*icol+ioffst-1, icol-1).CVector(), ilda, extra, dummy)
						}

						//                    Chase "EXTRA" back up
						ir = jr
						ic = icol
						for jch = jr - jkl; jch >= 1; jch -= (jkl + jku) {
							if ir < m {
								realc, s, dummy = golapack.Zlartg(a.Get(ir+1-iskew*(ic+1)+ioffst-1, ic), extra)
								dummy = Zlarnd(5, *iseed)
								c = cmplx.Conj(complex(realc, 0) * dummy)
								s = cmplx.Conj(-s * dummy)
							}
							irow = max(1, jch-jku)
							il = ir + 2 - irow
							ctemp = czero
							iltemp = jch > jku
							ctemp, extra = Zlarot(false, iltemp, true, il, c, s, a.Off(irow-iskew*ic+ioffst-1, ic-1).CVector(), ilda, ctemp, extra)
							if iltemp {
								realc, s, dummy = golapack.Zlartg(a.Get(irow+1-iskew*(ic+1)+ioffst-1, ic), ctemp)
								dummy = Zlarnd(5, *iseed)
								c = cmplx.Conj(complex(realc, 0) * dummy)
								s = cmplx.Conj(-s * dummy)

								icol = max(1, jch-jku-jkl)
								il = ic + 2 - icol
								extra = czero
								extra, ctemp = Zlarot(true, jch > jku+jkl, true, il, c, s, a.Off(irow-iskew*icol+ioffst-1, icol-1).CVector(), ilda, extra, ctemp)
								ic = icol
								ir = irow
							}
						}
					}
				}

				jku = uub
				for jkl = 1; jkl <= llb; jkl++ {
					//                 Transform from bandwidth JKL-1, JKU to JKL, JKU
					for jc = 1; jc <= min(n+jkl, m)+jku-1; jc++ {
						extra = czero
						angle = twopi * Dlarnd(1, iseed)
						c = complex(math.Cos(angle), 0) * Zlarnd(5, *iseed)
						s = complex(math.Sin(angle), 0) * Zlarnd(5, *iseed)
						irow = max(1, jc-jku)
						if jc < n {
							il = min(m, jc+jkl) + 1 - irow
							extra, dummy = Zlarot(false, jc > jku, false, il, c, s, a.Off(irow-iskew*jc+ioffst-1, jc-1).CVector(), ilda, extra, dummy)
						}

						//                    Chase "EXTRA" back up
						ic = jc
						ir = irow
						for jch = jc - jku; jch >= 1; jch -= (jkl + jku) {
							if ic < n {
								realc, s, dummy = golapack.Zlartg(a.Get(ir+1-iskew*(ic+1)+ioffst-1, ic), extra)
								dummy = Zlarnd(5, *iseed)
								c = cmplx.Conj(complex(realc, 0) * dummy)
								s = cmplx.Conj(-s * dummy)
							}
							icol = max(1, jch-jkl)
							il = ic + 2 - icol
							ctemp = czero
							iltemp = jch > jkl
							ctemp, extra = Zlarot(true, iltemp, true, il, c, s, a.Off(ir-iskew*icol+ioffst-1, icol-1).CVector(), ilda, ctemp, extra)
							if iltemp {
								realc, s, dummy = golapack.Zlartg(a.Get(ir+1-iskew*(icol+1)+ioffst-1, icol), ctemp)
								dummy = Zlarnd(5, *iseed)
								c = cmplx.Conj(complex(realc, 0) * dummy)
								s = cmplx.Conj(-s * dummy)
								irow = max(1, jch-jkl-jku)
								il = ir + 2 - irow
								extra = czero
								extra, ctemp = Zlarot(false, jch > jkl+jku, true, il, c, s, a.Off(irow-iskew*icol+ioffst-1, icol-1).CVector(), ilda, extra, ctemp)
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
					iendch = min(m, n+jkl) - 1
					for jc = min(m+jku, n) - 1; jc >= 1-jkl; jc-- {
						extra = czero
						angle = twopi * Dlarnd(1, iseed)
						c = complex(math.Cos(angle), 0) * Zlarnd(5, *iseed)
						s = complex(math.Sin(angle), 0) * Zlarnd(5, *iseed)
						irow = max(1, jc-jku+1)
						if jc > 0 {
							il = min(m, jc+jkl+1) + 1 - irow
							dummy, extra = Zlarot(false, false, jc+jkl < m, il, c, s, a.Off(irow-iskew*jc+ioffst-1, jc-1).CVector(), ilda, dummy, extra)
						}

						//                    Chase "EXTRA" back down
						ic = jc
						for jch = jc + jkl; jch <= iendch; jch += jkl + jku {
							ilextr = ic > 0
							if ilextr {
								realc, s, dummy = golapack.Zlartg(a.Get(jch-iskew*ic+ioffst-1, ic-1), extra)
								dummy = Zlarnd(5, *iseed)
								c = complex(realc, 0) * dummy
								s = s * dummy
							}
							ic = max(1, ic)
							icol = min(n-1, jch+jku)
							iltemp = jch+jku < n
							ctemp = czero
							extra, ctemp = Zlarot(true, ilextr, iltemp, icol+2-ic, c, s, a.Off(jch-iskew*ic+ioffst-1, ic-1).CVector(), ilda, extra, ctemp)
							if iltemp {
								realc, s, dummy = golapack.Zlartg(a.Get(jch-iskew*icol+ioffst-1, icol-1), ctemp)
								dummy = Zlarnd(5, *iseed)
								c = complex(realc, 0) * dummy
								s = s * dummy
								il = min(iendch, jch+jkl+jku) + 2 - jch
								extra = czero
								ctemp, extra = Zlarot(false, true, jch+jkl+jku <= iendch, il, c, s, a.Off(jch-iskew*icol+ioffst-1, icol-1).CVector(), ilda, ctemp, extra)
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
					iendch = min(n, m+jku) - 1
					for jr = min(n+jkl, m) - 1; jr >= 1-jku; jr-- {
						extra = czero
						angle = twopi * Dlarnd(1, iseed)
						c = complex(math.Cos(angle), 0) * Zlarnd(5, *iseed)
						s = complex(math.Sin(angle), 0) * Zlarnd(5, *iseed)
						icol = max(1, jr-jkl+1)
						if jr > 0 {
							il = min(n, jr+jku+1) + 1 - icol
							dummy, extra = Zlarot(true, false, jr+jku < n, il, c, s, a.Off(jr-iskew*icol+ioffst-1, icol-1).CVector(), ilda, dummy, extra)
						}

						//                    Chase "EXTRA" back down
						ir = jr
						for jch = jr + jku; jch <= iendch; jch += jkl + jku {
							ilextr = ir > 0
							if ilextr {
								realc, s, dummy = golapack.Zlartg(a.Get(ir-iskew*jch+ioffst-1, jch-1), extra)
								dummy = Zlarnd(5, *iseed)
								c = complex(realc, 0) * dummy
								s = s * dummy
							}
							ir = max(1, ir)
							irow = min(m-1, jch+jkl)
							iltemp = jch+jkl < m
							ctemp = czero
							extra, ctemp = Zlarot(false, ilextr, iltemp, irow+2-ir, c, s, a.Off(ir-iskew*jch+ioffst-1, jch-1).CVector(), ilda, extra, ctemp)
							if iltemp {
								realc, s, dummy = golapack.Zlartg(a.Get(irow-iskew*jch+ioffst-1, jch-1), ctemp)
								dummy = Zlarnd(5, *iseed)
								c = complex(realc, 0) * dummy
								s = s * dummy
								il = min(iendch, jch+jkl+jku) + 2 - jch
								extra = czero
								ctemp, extra = Zlarot(true, true, jch+jkl+jku <= iendch, il, c, s, a.Off(irow-iskew*jch+ioffst-1, jch-1).CVector(), ilda, ctemp, extra)
								ir = irow
							}
						}
					}
				}

			}

		} else {
			//           Symmetric -- A = U D U'
			//           Hermitian -- A = U D U*
			ipackg = ipack
			ioffg = ioffst

			if topdwn {
				//              Top-Down -- Generate Upper triangle only
				if ipack >= 5 {
					ipackg = 6
					ioffg = uub + 1
				} else {
					ipackg = 1
				}

				for j = 1; j <= mnmin; j++ {
					a.Set((1-iskew)*j+ioffg-1, j-1, complex(d.Get(j-1), 0))
				}

				for k = 1; k <= uub; k++ {
					for jc = 1; jc <= n-1; jc++ {
						irow = max(1, jc-k)
						il = min(jc+1, k+2)
						extra = czero
						ctemp = a.Get(jc-iskew*(jc+1)+ioffg-1, jc)
						angle = twopi * Dlarnd(1, iseed)
						c = complex(math.Cos(angle), 0) * Zlarnd(5, *iseed)
						s = complex(math.Sin(angle), 0) * Zlarnd(5, *iseed)
						if zsym {
							ct = c
							st = s
						} else {
							ctemp = cmplx.Conj(ctemp)
							ct = cmplx.Conj(c)
							st = cmplx.Conj(s)
						}
						extra, ctemp = Zlarot(false, jc > k, true, il, c, s, a.Off(irow-iskew*jc+ioffg-1, jc-1).CVector(), ilda, extra, ctemp)
						ctemp, dummy = Zlarot(true, true, false, min(k, n-jc)+1, ct, st, a.Off((1-iskew)*jc+ioffg-1, jc-1).CVector(), ilda, ctemp, dummy)

						//                    Chase EXTRA back up the matrix
						icol = jc
						for jch = jc - k; jch >= 1; jch -= k {
							realc, s, dummy = golapack.Zlartg(a.Get(jch+1-iskew*(icol+1)+ioffg-1, icol), extra)
							dummy = Zlarnd(5, *iseed)
							c = cmplx.Conj(complex(realc, 0) * dummy)
							s = cmplx.Conj(-s * dummy)
							ctemp = a.Get(jch-iskew*(jch+1)+ioffg-1, jch)
							if zsym {
								ct = c
								st = s
							} else {
								ctemp = cmplx.Conj(ctemp)
								ct = cmplx.Conj(c)
								st = cmplx.Conj(s)
							}
							ctemp, extra = Zlarot(true, true, true, k+2, c, s, a.Off((1-iskew)*jch+ioffg-1, jch-1).CVector(), ilda, ctemp, extra)
							irow = max(1, jch-k)
							il = min(jch+1, k+2)
							extra = czero
							extra, ctemp = Zlarot(false, jch > k, true, il, ct, st, a.Off(irow-iskew*jch+ioffg-1, jch-1).CVector(), ilda, extra, ctemp)
							icol = jch
						}
					}
				}

				//              If we need lower triangle, copy from upper. Note that
				//              the order of copying is chosen to work for 'q' -> 'b'
				if ipack != ipackg && ipack != 3 {
					for jc = 1; jc <= n; jc++ {
						irow = ioffst - iskew*jc
						if zsym {
							for jr = jc; jr <= min(n, jc+uub); jr++ {
								a.Set(jr+irow-1, jc-1, a.Get(jc-iskew*jr+ioffg-1, jr-1))
							}
						} else {
							for jr = jc; jr <= min(n, jc+uub); jr++ {
								a.Set(jr+irow-1, jc-1, cmplx.Conj(a.Get(jc-iskew*jr+ioffg-1, jr-1)))
							}
						}
					}
					if ipack == 5 {
						for jc = n - uub + 1; jc <= n; jc++ {
							for jr = n + 2 - jc; jr <= uub+1; jr++ {
								a.Set(jr-1, jc-1, czero)
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

				for j = 1; j <= mnmin; j++ {
					a.Set((1-iskew)*j+ioffg-1, j-1, complex(d.Get(j-1), 0))
				}

				for k = 1; k <= uub; k++ {
					for jc = n - 1; jc >= 1; jc-- {
						il = min(n+1-jc, k+2)
						extra = czero
						ctemp = a.Get(1+(1-iskew)*jc+ioffg-1, jc-1)
						angle = twopi * Dlarnd(1, iseed)
						c = complex(math.Cos(angle), 0) * Zlarnd(5, *iseed)
						s = complex(math.Sin(angle), 0) * Zlarnd(5, *iseed)
						if zsym {
							ct = c
							st = s
						} else {
							ctemp = cmplx.Conj(ctemp)
							ct = cmplx.Conj(c)
							st = cmplx.Conj(s)
						}
						ctemp, extra = Zlarot(false, true, n-jc > k, il, c, s, a.Off((1-iskew)*jc+ioffg-1, jc-1).CVector(), ilda, ctemp, extra)
						icol = max(1, jc-k+1)
						dummy, ctemp = Zlarot(true, false, true, jc+2-icol, ct, st, a.Off(jc-iskew*icol+ioffg-1, icol-1).CVector(), ilda, dummy, ctemp)

						//                    Chase EXTRA back down the matrix
						icol = jc
						for jch = jc + k; jch <= n-1; jch += k {
							realc, s, dummy = golapack.Zlartg(a.Get(jch-iskew*icol+ioffg-1, icol-1), extra)
							dummy = Zlarnd(5, *iseed)
							c = complex(realc, 0) * dummy
							s = s * dummy
							ctemp = a.Get(1+(1-iskew)*jch+ioffg-1, jch-1)
							if zsym {
								ct = c
								st = s
							} else {
								ctemp = cmplx.Conj(ctemp)
								ct = cmplx.Conj(c)
								st = cmplx.Conj(s)
							}
							extra, ctemp = Zlarot(true, true, true, k+2, c, s, a.Off(jch-iskew*icol+ioffg-1, icol-1).CVector(), ilda, extra, ctemp)
							il = min(n+1-jch, k+2)
							extra = czero
							ctemp, extra = Zlarot(false, true, n-jch > k, il, ct, st, a.Off((1-iskew)*jch+ioffg-1, jch-1).CVector(), ilda, ctemp, extra)
							icol = jch
						}
					}
				}

				//              If we need upper triangle, copy from lower. Note that
				//              the order of copying is chosen to work for 'b' -> 'q'
				if ipack != ipackg && ipack != 4 {
					for jc = n; jc >= 1; jc-- {
						irow = ioffst - iskew*jc
						if zsym {
							for jr = jc; jr >= max(1, jc-uub); jr-- {
								a.Set(jr+irow-1, jc-1, a.Get(jc-iskew*jr+ioffg-1, jr-1))
							}
						} else {
							for jr = jc; jr >= max(1, jc-uub); jr-- {
								a.Set(jr+irow-1, jc-1, cmplx.Conj(a.Get(jc-iskew*jr+ioffg-1, jr-1)))
							}
						}
					}
					if ipack == 6 {
						for jc = 1; jc <= uub; jc++ {
							for jr = 1; jr <= uub+1-jc; jr++ {
								a.Set(jr-1, jc-1, czero)
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

			//           Ensure that the diagonal is real if Hermitian
			if !zsym {
				for jc = 1; jc <= n; jc++ {
					irow = ioffst + (1-iskew)*jc
					a.Set(irow-1, jc-1, complex(real(a.Get(irow-1, jc-1)), 0))
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
			err = Zlagge(mr, nc, llb, uub, d, a, iseed, work)
		} else {
			//           Symmetric -- A = U D U' or
			//           Hermitian -- A = U D U*
			if zsym {
				err = Zlagsy(m, llb, d, a, iseed, work)
			} else {
				err = Zlaghe(m, llb, d, a, iseed, work)
			}
		}

		if iinfo != 0 {
			err = fmt.Errorf("Error return from Zlagge, Zlaghe or Zlagsy")
			return
		}
	}

	//     5)      Pack the matrix
	if ipack != ipackg {
		if ipack == 1 {
			//           'U' -- Upper triangular, not packed
			for j = 1; j <= m; j++ {
				for i = j + 1; i <= m; i++ {
					a.Set(i-1, j-1, czero)
				}
			}

		} else if ipack == 2 {
			//           'L' -- Lower triangular, not packed
			for j = 2; j <= m; j++ {
				for i = 1; i <= j-1; i++ {
					a.Set(i-1, j-1, czero)
				}
			}

		} else if ipack == 3 {
			//           'C' -- Upper triangle packed Columnwise.
			icol = 1
			irow = 0
			for j = 1; j <= m; j++ {
				for i = 1; i <= j; i++ {
					irow = irow + 1
					if irow > a.Rows {
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
			for j = 1; j <= m; j++ {
				for i = j; i <= m; i++ {
					irow = irow + 1
					if irow > a.Rows {
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
				for i = min(j+llb, m); i >= 1; i-- {
					a.Set(i-j+uub, j-1, a.Get(i-1, j-1))
				}
			}

			for j = uub + 2; j <= n; j++ {
				for i = j - uub; i <= min(j+llb, m); i++ {
					a.Set(i-j+uub, j-1, a.Get(i-1, j-1))
				}
			}
		}

		//        If packed, zero out extraneous elements.
		//
		//        Symmetric/Triangular Packed --
		//        zero out everything after A(IROW,ICOL)
		if ipack == 3 || ipack == 4 {
			for jc = icol; jc <= m; jc++ {
				for jr = irow + 1; jr <= a.Rows; jr++ {
					a.Set(jr-1, jc-1, czero)
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
			ir2 = uub + m + 2
			for jc = 1; jc <= n; jc++ {
				for jr = 1; jr <= uub+1-jc; jr++ {
					a.Set(jr-1, jc-1, czero)
				}
				for jr = max(1, min(ir1, ir2-jc)); jr <= a.Rows; jr++ {
					a.Set(jr-1, jc-1, czero)
				}
			}
		}
	}

	return
}
