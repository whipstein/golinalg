package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgees computes for an N-by-N real nonsymmetric matrix A, the
// eigenvalues, the real Schur form T, and, optionally, the matrix of
// Schur vectors Z.  This gives the Schur factorization A = Z*T*(Z**T).
//
// Optionally, it also orders the eigenvalues on the diagonal of the
// real Schur form so that selected eigenvalues are at the top left.
// The leading columns of Z then form an orthonormal basis for the
// invariant subspace corresponding to the selected eigenvalues.
//
// A matrix is in real Schur form if it is upper quasi-triangular with
// 1-by-1 and 2-by-2 blocks. 2-by-2 blocks will be standardized in the
// form
//         [  a  b  ]
//         [  c  a  ]
//
// where b*c < 0. The eigenvalues of such a block are a +- math.Sqrt(bc).
func Dgees(jobvs, sort byte, _select dslectFunc, n int, a *mat.Matrix, wr, wi *mat.Vector, vs *mat.Matrix, work *mat.Vector, lwork int, bwork *[]bool) (sdim, info int, err error) {
	var cursl, lastsl, lquery, lst2sl, scalea, wantst, wantvs bool
	var anrm, bignum, cscale, eps, one, s, sep, smlnum, zero float64
	var hswork, i, i1, i2, ibal, icond, ieval, ihi, ilo, inxt, ip, itau, iwrk, maxwrk, minwrk int

	dum := vf(1)
	idum := make([]int, 1)

	zero = 0.0
	one = 1.0

	//     Test the input arguments
	lquery = (lwork == -1)
	wantvs = jobvs == 'V'
	wantst = sort == 'S'
	if (!wantvs) && jobvs != 'N' {
		err = fmt.Errorf("(!wantvs) && jobvs != 'N': jobvs='%c'", jobvs)
	} else if (!wantst) && sort != 'N' {
		err = fmt.Errorf("(!wantst) && sort != 'N': sort='%c'", sort)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if vs.Rows < 1 || (wantvs && vs.Rows < n) {
		err = fmt.Errorf("vs.Rows < 1 || (wantvs && vs.Rows < n): jobvs='%c', vs.Rows=%v, n=%v", jobvs, vs.Rows, n)
	}

	//     Compute workspace
	//      (Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of workspace needed at that point in the code,
	//       as well as the preferred amount for good performance.
	//       NB refers to the optimal block size for the immediately
	//       following subroutine, as returned by ILAENV.
	//       HSWORK refers to the workspace preferred by DHSEQR, as
	//       calculated below. HSWORK is computed assuming ILO=1 and IHI=N,
	//       the worst case.)
	if err == nil {
		if n == 0 {
			minwrk = 1
			maxwrk = 1
		} else {
			maxwrk = 2*n + n*Ilaenv(1, "Dgehrd", []byte{' '}, n, 1, n, 0)
			minwrk = 3 * n

			if ieval, err = Dhseqr('S', jobvs, n, 1, n, a, wr, wi, vs, work, -1); err != nil {
				panic(err)
			}
			hswork = int(work.Get(0))

			if !wantvs {
				maxwrk = max(maxwrk, n+hswork)
			} else {
				maxwrk = max(maxwrk, 2*n+(n-1)*Ilaenv(1, "Dorghr", []byte{' '}, n, 1, n, -1))
				maxwrk = max(maxwrk, n+hswork)
			}
		}
		work.Set(0, float64(maxwrk))

		if lwork < minwrk && !lquery {
			err = fmt.Errorf("lwork < minwrk && !lquery: lwork=%v, minwrk=%v, lquery=%v", lwork, minwrk, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dgees", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		sdim = 0
		return
	}

	//     Get machine constants
	eps = Dlamch(Precision)
	smlnum = Dlamch(SafeMinimum)
	bignum = one / smlnum
	smlnum, bignum = Dlabad(smlnum, bignum)
	smlnum = math.Sqrt(smlnum) / eps
	bignum = one / smlnum

	//     Scale A if max element outside range [SMLNUM,BIGNUM]
	anrm = Dlange('M', n, n, a, dum)
	scalea = false
	if anrm > zero && anrm < smlnum {
		scalea = true
		cscale = smlnum
	} else if anrm > bignum {
		scalea = true
		cscale = bignum
	}
	if scalea {
		if err = Dlascl('G', 0, 0, anrm, cscale, n, n, a); err != nil {
			panic(err)
		}
	}

	//     Permute the matrix to make it more nearly triangular
	//     (Workspace: need N)
	ibal = 1
	if ilo, ihi, err = Dgebal('P', n, a, work.Off(ibal-1)); err != nil {
		panic(err)
	}

	//     Reduce to upper Hessenberg form
	//     (Workspace: need 3*N, prefer 2*N+N*NB)
	itau = n + ibal
	iwrk = n + itau
	if err = Dgehrd(n, ilo, ihi, a, work.Off(itau-1), work.Off(iwrk-1), lwork-iwrk+1); err != nil {
		panic(err)
	}

	if wantvs {
		//        Copy Householder vectors to VS
		Dlacpy(Lower, n, n, a, vs)

		//        Generate orthogonal matrix in VS
		//        (Workspace: need 3*N-1, prefer 2*N+(N-1)*NB)
		if err = Dorghr(n, ilo, ihi, vs, work.Off(itau-1), work.Off(iwrk-1), lwork-iwrk+1); err != nil {
			panic(err)
		}
	}

	sdim = 0

	//     Perform QR iteration, accumulating Schur vectors in VS if desired
	//     (Workspace: need N+1, prefer N+HSWORK (see comments) )
	iwrk = itau
	if ieval, err = Dhseqr('S', jobvs, n, ilo, ihi, a, wr, wi, vs, work.Off(iwrk-1), lwork-iwrk+1); err != nil {
		panic(err)
	}
	if ieval > 0 {
		info = ieval
	}

	//     Sort eigenvalues if desired
	if wantst && info == 0 {
		if scalea {
			if err = Dlascl('G', 0, 0, cscale, anrm, n, 1, wr.Matrix(n, opts)); err != nil {
				panic(err)
			}
			if err = Dlascl('G', 0, 0, cscale, anrm, n, 1, wi.Matrix(n, opts)); err != nil {
				panic(err)
			}
		}
		for i = 1; i <= n; i++ {
			(*bwork)[i-1] = _select(wr.GetPtr(i-1), wi.GetPtr(i-1))
		}

		//        Reorder eigenvalues and transform Schur vectors
		//        (Workspace: none needed)
		if sdim, s, sep, icond, err = Dtrsen('N', jobvs, *bwork, n, a, vs, wr, wi, s, sep, work.Off(iwrk-1), lwork-iwrk+1, &idum, 1); icond > 0 {
			info = n + icond
		}
	}

	if wantvs {
		//        Undo balancing
		//        (Workspace: need N)
		if err = Dgebak('P', Right, n, ilo, ihi, work.Off(ibal-1), n, vs); err != nil {
			panic(err)
		}
	}

	if scalea {
		//        Undo scaling for the Schur form of A
		if err = Dlascl('H', 0, 0, cscale, anrm, n, n, a); err != nil {
			panic(err)
		}
		wr.Copy(n, a.OffIdx(0).Vector(), a.Rows+1, 1)
		if cscale == smlnum {
			//           If scaling back towards underflow, adjust WI if an
			//           offdiagonal element of a 2-by-2 block in the Schur form
			//           underflows.
			if ieval > 0 {
				i1 = ieval + 1
				i2 = ihi - 1
				if err = Dlascl('G', 0, 0, cscale, anrm, ilo-1, 1, wi.Matrix(max(ilo-1, 1), opts)); err != nil {
					panic(err)
				}
			} else if wantst {
				i1 = 1
				i2 = n - 1
			} else {
				i1 = ilo
				i2 = ihi - 1
			}
			inxt = i1 - 1
			for i = i1; i <= i2; i++ {
				if i < inxt {
					goto label20
				}
				if wi.Get(i-1) == zero {
					inxt = i + 1
				} else {
					if a.Get(i, i-1) == zero {
						wi.Set(i-1, zero)
						wi.Set(i, zero)
					} else if a.Get(i, i-1) != zero && a.Get(i-1, i) == zero {
						wi.Set(i-1, zero)
						wi.Set(i, zero)
						if i > 1 {
							a.Off(0, i).Vector().Swap(i-1, a.Off(0, i-1).Vector(), 1, 1)
						}
						if n > i+1 {
							a.Off(i, i+2-1).Vector().Swap(n-i-1, a.Off(i-1, i+2-1).Vector(), a.Rows, a.Rows)
						}
						if wantvs {
							vs.Off(0, i).Vector().Swap(n, vs.Off(0, i-1).Vector(), 1, 1)
						}
						a.Set(i-1, i, a.Get(i, i-1))
						a.Set(i, i-1, zero)
					}
					inxt = i + 2
				}
			label20:
			}
		}

		//        Undo scaling for the imaginary part of the eigenvalues
		if err = Dlascl('G', 0, 0, cscale, anrm, n-ieval, 1, wi.Off(ieval).Matrix(max(n-ieval, 1), opts)); err != nil {
			panic(err)
		}
	}

	if wantst && info == 0 {
		//        Check if reordering successful
		lastsl = true
		lst2sl = true
		sdim = 0
		ip = 0
		for i = 1; i <= n; i++ {
			cursl = _select(wr.GetPtr(i-1), wi.GetPtr(i-1))
			if wi.Get(i-1) == zero {
				if cursl {
					sdim = sdim + 1
				}
				ip = 0
				if cursl && !lastsl {
					info = n + 2
				}
			} else {
				if ip == 1 {
					//                 Last eigenvalue of conjugate pair
					cursl = cursl || lastsl
					lastsl = cursl
					if cursl {
						sdim = sdim + 2
					}
					ip = -1
					if cursl && !lst2sl {
						info = n + 2
					}
				} else {
					//                 First eigenvalue of conjugate pair
					ip = 1
				}
			}
			lst2sl = lastsl
			lastsl = cursl
		}
	}

	work.Set(0, float64(maxwrk))

	return
}
