package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgeev computes for an N-by-N complex nonsymmetric matrix A, the
// eigenvalues and, optionally, the left and/or right eigenvectors.
//
// The right eigenvector v(j) of A satisfies
//                  A * v(j) = lambda(j) * v(j)
// where lambda(j) is its eigenvalue.
// The left eigenvector u(j) of A satisfies
//               u(j)**H * A = lambda(j) * u(j)**H
// where u(j)**H denotes the conjugate transpose of u(j).
//
// The computed eigenvectors are normalized to have Euclidean norm
// equal to 1 and largest component real.
func Zgeev(jobvl, jobvr byte, n int, a *mat.CMatrix, w *mat.CVector, vl, vr *mat.CMatrix, work *mat.CVector, lwork int, rwork *mat.Vector) (info int, err error) {
	var lquery, scalea, wantvl, wantvr bool
	var side mat.MatSide
	var tmp complex128
	var anrm, bignum, cscale, eps, one, scl, smlnum, zero float64
	var hswork, i, ibal, ihi, ilo, irwork, itau, iwrk, k, lworkTrevc, maxwrk, minwrk int

	_select := make([]bool, 1)
	dum := vf(1)

	zero = 0.0
	one = 1.0

	//     Test the input arguments
	lquery = (lwork == -1)
	wantvl = jobvl == 'V'
	wantvr = jobvr == 'V'
	if (!wantvl) && (jobvl != 'N') {
		err = fmt.Errorf("(!wantvl) && (jobvl != 'N'): jobvl='%c'", jobvl)
	} else if (!wantvr) && (jobvr != 'N') {
		err = fmt.Errorf("(!wantvr) && (jobvr != 'N'): jobvr='%c'", jobvr)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if vl.Rows < 1 || (wantvl && vl.Rows < n) {
		err = fmt.Errorf("vl.Rows < 1 || (wantvl && vl.Rows < n): jobvl='%c', vl.Rows=%v, n=%v", jobvl, vl.Rows, n)
	} else if vr.Rows < 1 || (wantvr && vr.Rows < n) {
		err = fmt.Errorf("vr.Rows < 1 || (wantvr && vr.Rows < n): jobvr='%c', vr.Rows=%v, n=%v", jobvr, vr.Rows, n)
	}

	//     Compute workspace
	//      (Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of workspace needed at that point in the code,
	//       as well as the preferred amount for good performance.
	//       CWorkspace refers to complex workspace, and RWorkspace to real
	//       workspace. NB refers to the optimal block size for the
	//       immediately following subroutine, as returned by ILAENV.
	//       HSWORK refers to the workspace preferred by ZHSEQR, as
	//       calculated below. HSWORK is computed assuming ILO=1 and IHI=N,
	//       the worst case.)
	if err == nil {
		if n == 0 {
			minwrk = 1
			maxwrk = 1
		} else {
			maxwrk = n + n*Ilaenv(1, "Zgehrd", []byte{' '}, n, 1, n, 0)
			minwrk = 2 * n
			if wantvl {
				maxwrk = max(maxwrk, n+(n-1)*Ilaenv(1, "Zunghr", []byte{' '}, n, 1, n, -1))
				if _, err = Ztrevc3(Left, 'B', _select, n, a, vl, vr, n, work, -1, rwork, -1); err != nil {
					panic(err)
				}
				lworkTrevc = int(work.GetRe(0))
				maxwrk = max(maxwrk, n+lworkTrevc)
				if info, err = Zhseqr('S', 'V', n, 1, n, a, w, vl, work, -1); err != nil {
					panic(err)
				}
			} else if wantvr {
				maxwrk = max(maxwrk, n+(n-1)*Ilaenv(1, "Zunghr", []byte{' '}, n, 1, n, -1))
				if _, err = Ztrevc3(Right, 'B', _select, n, a, vl, vr, n, work, -1, rwork, -1); err != nil {
					panic(err)
				}
				lworkTrevc = int(work.GetRe(0))
				maxwrk = max(maxwrk, n+lworkTrevc)
				if info, err = Zhseqr('S', 'V', n, 1, n, a, w, vr, work, -1); err != nil {
					panic(err)
				}
			} else {
				if info, err = Zhseqr('E', 'N', n, 1, n, a, w, vr, work, -1); err != nil {
					panic(err)
				}
			}
			hswork = int(work.GetRe(0))
			maxwrk = max(maxwrk, hswork, minwrk)
		}
		work.SetRe(0, float64(maxwrk))
		//
		if lwork < minwrk && !lquery {
			err = fmt.Errorf("lwork < minwrk && !lquery: lwork=%v, minwrk=%v, lquery=%v", lwork, minwrk, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Zgeev", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
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
	anrm = Zlange('M', n, n, a, dum)
	scalea = false
	if anrm > zero && anrm < smlnum {
		scalea = true
		cscale = smlnum
	} else if anrm > bignum {
		scalea = true
		cscale = bignum
	}
	if scalea {
		if err = Zlascl('G', 0, 0, anrm, cscale, n, n, a); err != nil {
			panic(err)
		}
	}

	//     Balance the matrix
	//     (CWorkspace: none)
	//     (RWorkspace: need N)
	ibal = 1
	if ilo, ihi, err = Zgebal('B', n, a, rwork.Off(ibal-1)); err != nil {
		panic(err)
	}

	//     Reduce to upper Hessenberg form
	//     (CWorkspace: need 2*N, prefer N+N*NB)
	//     (RWorkspace: none)
	itau = 1
	iwrk = itau + n
	if err = Zgehrd(n, ilo, ihi, a, work.Off(itau-1), work.Off(iwrk-1), lwork-iwrk+1); err != nil {
		panic(err)
	}

	if wantvl {
		//        Want left eigenvectors
		//        Copy Householder vectors to VL
		side = Left
		Zlacpy(Lower, n, n, a, vl)

		//        Generate unitary matrix in VL
		//        (CWorkspace: need 2*N-1, prefer N+(N-1)*NB)
		//        (RWorkspace: none)
		if err = Zunghr(n, ilo, ihi, vl, work.Off(itau-1), work.Off(iwrk-1), lwork-iwrk+1); err != nil {
			panic(err)
		}

		//        Perform QR iteration, accumulating Schur vectors in VL
		//        (CWorkspace: need 1, prefer HSWORK (see comments) )
		//        (RWorkspace: none)
		iwrk = itau
		if info, err = Zhseqr('S', 'V', n, ilo, ihi, a, w, vl, work.Off(iwrk-1), lwork-iwrk+1); err != nil {
			panic(err)
		}

		if wantvr {
			//           Want left and right eigenvectors
			//           Copy Schur vectors to VR
			side = Both
			Zlacpy(Full, n, n, vl, vr)
		}

	} else if wantvr {
		//        Want right eigenvectors
		//        Copy Householder vectors to VR
		side = Right
		Zlacpy(Lower, n, n, a, vr)

		//        Generate unitary matrix in VR
		//        (CWorkspace: need 2*N-1, prefer N+(N-1)*NB)
		//        (RWorkspace: none)
		if err = Zunghr(n, ilo, ihi, vr, work.Off(itau-1), work.Off(iwrk-1), lwork-iwrk+1); err != nil {
			panic(err)
		}

		//        Perform QR iteration, accumulating Schur vectors in VR
		//        (CWorkspace: need 1, prefer HSWORK (see comments) )
		//        (RWorkspace: none)
		iwrk = itau
		if info, err = Zhseqr('S', 'V', n, ilo, ihi, a, w, vr, work.Off(iwrk-1), lwork-iwrk+1); err != nil {
			panic(err)
		}

	} else {
		//        Compute eigenvalues only
		//        (CWorkspace: need 1, prefer HSWORK (see comments) )
		//        (RWorkspace: none)
		iwrk = itau
		if info, err = Zhseqr('E', 'N', n, ilo, ihi, a, w, vr, work.Off(iwrk-1), lwork-iwrk+1); err != nil {
			panic(err)
		}
	}

	//     If INFO .NE. 0 from ZHSEQR, then quit
	if info != 0 {
		goto label50
	}

	if wantvl || wantvr {
		//        Compute left and/or right eigenvectors
		//        (CWorkspace: need 2*N, prefer N + 2*N*NB)
		//        (RWorkspace: need 2*N)
		irwork = ibal + n
		if _, err = Ztrevc3(side, 'B', _select, n, a, vl, vr, n, work.Off(iwrk-1), lwork-iwrk+1, rwork.Off(irwork-1), n); err != nil {
			panic(err)
		}
	}

	if wantvl {
		//        Undo balancing of left eigenvectors
		//        (CWorkspace: none)
		//        (RWorkspace: need N)
		if err = Zgebak('B', Left, n, ilo, ihi, rwork.Off(ibal-1), n, vl); err != nil {
			panic(err)
		}

		//        Normalize left eigenvectors and make largest component real
		for i = 1; i <= n; i++ {
			scl = one / vl.Off(0, i-1).CVector().Nrm2(n, 1)
			vl.Off(0, i-1).CVector().Dscal(n, scl, 1)
			for k = 1; k <= n; k++ {
				rwork.Set(irwork+k-1-1, math.Pow(vl.GetRe(k-1, i-1), 2)+math.Pow(vl.GetIm(k-1, i-1), 2))
			}
			k = rwork.Off(irwork-1).Iamax(n, 1)
			tmp = vl.GetConj(k-1, i-1) / complex(math.Sqrt(rwork.Get(irwork+k-1-1)), 0)
			vl.Off(0, i-1).CVector().Scal(n, tmp, 1)
			vl.Set(k-1, i-1, vl.GetReCmplx(k-1, i-1))
		}
	}

	if wantvr {
		//        Undo balancing of right eigenvectors
		//        (CWorkspace: none)
		//        (RWorkspace: need N)
		if err = Zgebak('B', Right, n, ilo, ihi, rwork.Off(ibal-1), n, vr); err != nil {
			panic(err)
		}

		//        Normalize right eigenvectors and make largest component real
		for i = 1; i <= n; i++ {
			scl = one / vr.Off(0, i-1).CVector().Nrm2(n, 1)
			vr.Off(0, i-1).CVector().Dscal(n, scl, 1)
			for k = 1; k <= n; k++ {
				rwork.Set(irwork+k-1-1, math.Pow(vr.GetRe(k-1, i-1), 2)+math.Pow(vr.GetIm(k-1, i-1), 2))
			}
			k = rwork.Off(irwork-1).Iamax(n, 1)
			tmp = vr.GetConj(k-1, i-1) / complex(math.Sqrt(rwork.Get(irwork+k-1-1)), 0)
			vr.Off(0, i-1).CVector().Scal(n, tmp, 1)
			vr.Set(k-1, i-1, vr.GetReCmplx(k-1, i-1))
		}
	}

	//     Undo scaling if necessary
label50:
	;
	if scalea {
		if err = Zlascl('G', 0, 0, cscale, anrm, n-info, 1, w.Off(info).CMatrix(max(n-info, 1), opts)); err != nil {
			panic(err)
		}
		if info > 0 {
			if err = Zlascl('G', 0, 0, cscale, anrm, ilo-1, 1, w.CMatrix(n, opts)); err != nil {
				panic(err)
			}
		}
	}

	work.SetRe(0, float64(maxwrk))

	return
}
