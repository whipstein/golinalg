package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgeesx computes for an N-by-N complex nonsymmetric matrix A, the
// eigenvalues, the Schur form T, and, optionally, the matrix of Schur
// vectors Z.  This gives the Schur factorization A = Z*T*(Z**H).
//
// Optionally, it also orders the eigenvalues on the diagonal of the
// Schur form so that selected eigenvalues are at the top left;
// computes a reciprocal condition number for the average of the
// selected eigenvalues (RCONDE); and computes a reciprocal condition
// number for the right invariant subspace corresponding to the
// selected eigenvalues (RCONDV).  The leading columns of Z form an
// orthonormal basis for this invariant subspace.
//
// For further explanation of the reciprocal condition numbers RCONDE
// and RCONDV, see Section 4.10 of the LAPACK Users' Guide (where
// these quantities are called s and sep respectively).
//
// A complex matrix is in Schur form if it is upper triangular.
func Zgeesx(jobvs, sort byte, _select func(complex128) bool, sense byte, n int, a *mat.CMatrix, w *mat.CVector, vs *mat.CMatrix, work *mat.CVector, lwork int, rwork *mat.Vector, bwork *[]bool) (sdim int, rconde, rcondv float64, info int, err error) {
	var lquery, scalea, wantsb, wantse, wantsn, wantst, wantsv, wantvs bool
	var anrm, bignum, cscale, eps, one, smlnum, zero float64
	var hswork, i, ibal, icond, ieval, ihi, ilo, itau, iwrk, lwrk, maxwrk, minwrk int

	dum := vf(1)

	zero = 0.0
	one = 1.0

	//     Test the input arguments
	wantvs = jobvs == 'V'
	wantst = sort == 'S'
	wantsn = sense == 'N'
	wantse = sense == 'E'
	wantsv = sense == 'V'
	wantsb = sense == 'B'
	lquery = (lwork == -1)

	if (!wantvs) && (jobvs != 'N') {
		err = fmt.Errorf("(!wantvs) && (jobvs != 'N'): jobvs='%c'", jobvs)
	} else if (!wantst) && (sort != 'N') {
		err = fmt.Errorf("(!wantst) && (sort != 'N'): sort='%c'", sort)
	} else if !(wantsn || wantse || wantsv || wantsb) || (!wantst && !wantsn) {
		err = fmt.Errorf("!(wantsn || wantse || wantsv || wantsb) || (!wantst && !wantsn): sense='%c'", sense)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if vs.Rows < 1 || (wantvs && vs.Rows < n) {
		err = fmt.Errorf("vs.Rows < 1 || (wantvs && vs.Rows < n): jobvs='%c', vs.Rows=%v, n=%v", jobvs, vs.Rows, n)
	}

	//     Compute workspace
	//      (Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of real workspace needed at that point in the
	//       code, as well as the preferred amount for good performance.
	//       CWorkspace refers to complex workspace, and RWorkspace to real
	//       workspace. NB refers to the optimal block size for the
	//       immediately following subroutine, as returned by ILAENV.
	//       HSWORK refers to the workspace preferred by ZHSEQR, as
	//       calculated below. HSWORK is computed assuming ILO=1 and IHI=N,
	//       the worst case.
	//       If SENSE = 'E', 'V' or 'B', then the amount of workspace needed
	//       depends on SDIM, which is computed by the routine ZTRSEN later
	//       in the code.)
	if err == nil {
		if n == 0 {
			minwrk = 1
			lwrk = 1
		} else {
			maxwrk = n + n*Ilaenv(1, "Zgehrd", []byte{' '}, n, 1, n, 0)
			minwrk = 2 * n

			if ieval, err = Zhseqr('S', jobvs, n, 1, n, a, w, vs, work, -1); err != nil {
				panic(err)
			}
			hswork = int(work.GetRe(0))

			if !wantvs {
				maxwrk = max(maxwrk, hswork)
			} else {
				maxwrk = max(maxwrk, n+(n-1)*Ilaenv(1, "Zunghr", []byte{' '}, n, 1, n, -1))
				maxwrk = max(maxwrk, hswork)
			}
			lwrk = maxwrk
			if !wantsn {
				lwrk = max(lwrk, (n*n)/2)
			}
		}
		work.SetRe(0, float64(lwrk))

		if lwork < minwrk && !lquery {
			err = fmt.Errorf("lwork < minwrk && !lquery: lwork=%v, minwrk=%v, lquery=%v", lwork, minwrk, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Zgeesx", err)
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

	//     Permute the matrix to make it more nearly triangular
	//     (CWorkspace: none)
	//     (RWorkspace: need N)
	ibal = 1
	if ilo, ihi, err = Zgebal('P', n, a, rwork.Off(ibal-1)); err != nil {
		panic(err)
	}

	//     Reduce to upper Hessenberg form
	//     (CWorkspace: need 2*N, prefer N+N*NB)
	//     (RWorkspace: none)
	itau = 1
	iwrk = n + itau
	if err = Zgehrd(n, ilo, ihi, a, work.Off(itau-1), work.Off(iwrk-1), lwork-iwrk+1); err != nil {
		panic(err)
	}

	if wantvs {
		//        Copy Householder vectors to VS
		Zlacpy(Lower, n, n, a, vs)

		//        Generate unitary matrix in VS
		//        (CWorkspace: need 2*N-1, prefer N+(N-1)*NB)
		//        (RWorkspace: none)
		if err = Zunghr(n, ilo, ihi, vs, work.Off(itau-1), work.Off(iwrk-1), lwork-iwrk+1); err != nil {
			panic(err)
		}
	}

	sdim = 0

	//     Perform QR iteration, accumulating Schur vectors in VS if desired
	//     (CWorkspace: need 1, prefer HSWORK (see comments) )
	//     (RWorkspace: none)
	iwrk = itau
	if ieval, err = Zhseqr('S', jobvs, n, ilo, ihi, a, w, vs, work.Off(iwrk-1), lwork-iwrk+1); err != nil {
		panic(err)
	}
	if ieval > 0 {
		info = ieval
	}

	//     Sort eigenvalues if desired
	if wantst && info == 0 {
		if scalea {
			if err = Zlascl('G', 0, 0, cscale, anrm, n, 1, w.CMatrix(n, opts)); err != nil {
				panic(err)
			}
		}
		for i = 1; i <= n; i++ {
			(*bwork)[i-1] = _select(w.Get(i - 1))
		}

		//        Reorder eigenvalues, transform Schur vectors, and compute
		//        reciprocal condition numbers
		//        (CWorkspace: if SENSE is not 'N', need 2*SDIM*(N-SDIM)
		//                     otherwise, need none )
		//        (RWorkspace: none)
		if sdim, rconde, rcondv, err = Ztrsen(sense, jobvs, *bwork, n, a, vs, w, work.Off(iwrk-1), lwork-iwrk+1); err != nil {
			panic(err)
		}
		if !wantsn {
			maxwrk = max(maxwrk, 2*sdim*(n-sdim))
		}
		if icond == -14 {
			//           Not enough complex workspace
			err = fmt.Errorf("icond == -14")
		}
	}

	if wantvs {
		//        Undo balancing
		//        (CWorkspace: none)
		//        (RWorkspace: need N)
		if err = Zgebak('P', Right, n, ilo, ihi, rwork.Off(ibal-1), n, vs); err != nil {
			panic(err)
		}
	}

	if scalea {
		//        Undo scaling for the Schur form of A
		if err = Zlascl('U', 0, 0, cscale, anrm, n, n, a); err != nil {
			panic(err)
		}
		goblas.Zcopy(n, a.CVector(0, 0, a.Rows+1), w.Off(0, 1))
		if (wantsv || wantsb) && info == 0 {
			dum.Set(0, rcondv)
			if err = Dlascl('G', 0, 0, cscale, anrm, 1, 1, dum.Matrix(1, opts)); err != nil {
				panic(err)
			}
			rcondv = dum.Get(0)
		}
	}

	work.SetRe(0, float64(maxwrk))

	return
}
