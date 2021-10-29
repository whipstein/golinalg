package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgges3 computes for a pair of N-by-N real nonsymmetric matrices (A,B),
// the generalized eigenvalues, the generalized real Schur form (S,T),
// optionally, the left and/or right matrices of Schur vectors (VSL and
// VSR). This gives the generalized Schur factorization
//
//          (A,B) = ( (VSL)*S*(VSR)**T, (VSL)*T*(VSR)**T )
//
// Optionally, it also orders the eigenvalues so that a selected cluster
// of eigenvalues appears in the leading diagonal blocks of the upper
// quasi-triangular matrix S and the upper triangular matrix T.The
// leading columns of VSL and VSR then form an orthonormal basis for the
// corresponding left and right eigenspaces (deflating subspaces).
//
// (If only the generalized eigenvalues are needed, use the driver
// DGGEV instead, which is faster.)
//
// A generalized eigenvalue for a pair of matrices (A,B) is a scalar w
// or a ratio alpha/beta = w, such that  A - w*B is singular.  It is
// usually represented as the pair (alpha,beta), as there is a
// reasonable interpretation for beta=0 or both being zero.
//
// A pair of matrices (S,T) is in generalized real Schur form if T is
// upper triangular with non-negative diagonal and S is block upper
// triangular with 1-by-1 and 2-by-2 blocks.  1-by-1 blocks correspond
// to real generalized eigenvalues, while 2-by-2 blocks of S will be
// "standardized" by making the corresponding elements of T have the
// form:
//         [  a  0  ]
//         [  0  b  ]
//
// and the pair of corresponding 2-by-2 blocks in S and T will have a
// complex conjugate pair of generalized eigenvalues.
func Dgges3(jobvsl, jobvsr, sort byte, selctg dlctesFunc, n int, a, b *mat.Matrix, alphar, alphai, beta *mat.Vector, vsl, vsr *mat.Matrix, work *mat.Vector, lwork int, bwork *[]bool) (sdim, info int, err error) {
	var cursl, ilascl, ilbscl, ilvsl, ilvsr, lastsl, lquery, lst2sl, wantst bool
	var anrm, anrmto, bignum, bnrm, bnrmto, eps, one, safmax, safmin, smlnum, zero float64
	var i, icols, ierr, ihi, ijobvl, ijobvr, ileft, ilo, ip, iright, irows, itau, iwrk, lwkopt int

	dif := vf(2)
	idum := make([]int, 1)

	zero = 0.0
	one = 1.0

	//     Decode the input arguments
	if jobvsl == 'N' {
		ijobvl = 1
		ilvsl = false
	} else if jobvsl == 'V' {
		ijobvl = 2
		ilvsl = true
	} else {
		ijobvl = -1
		ilvsl = false
	}

	if jobvsr == 'N' {
		ijobvr = 1
		ilvsr = false
	} else if jobvsr == 'V' {
		ijobvr = 2
		ilvsr = true
	} else {
		ijobvr = -1
		ilvsr = false
	}

	wantst = sort == 'S'

	//     Test the input arguments
	lquery = (lwork == -1)
	if ijobvl <= 0 {
		err = fmt.Errorf("ijobvl <= 0: jobvsl='%c'", jobvsl)
	} else if ijobvr <= 0 {
		err = fmt.Errorf("ijobvr <= 0: jobvsr='%c'", jobvsr)
	} else if (!wantst) && sort != 'N' {
		err = fmt.Errorf("(!wantst) && sort != 'N': sort='%c'", sort)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	} else if vsl.Rows < 1 || (ilvsl && vsl.Rows < n) {
		err = fmt.Errorf("vsl.Rows < 1 || (ilvsl && vsl.Rows < n): vsl.Rows=%v, ilvsl=%v, n=%v", vsl.Rows, ilvsl, n)
	} else if vsr.Rows < 1 || (ilvsr && vsr.Rows < n) {
		err = fmt.Errorf("vsr.Rows < 1 || (ilvsr && vsr.Rows < n): vsr.Rows=%v, ilvsr=%v, n=%v", vsr.Rows, ilvsr, n)
	} else if lwork < 6*n+16 && !lquery {
		err = fmt.Errorf("lwork < 6*n+16 && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
	}

	//     Compute workspace
	if err == nil {
		if err = Dgeqrf(n, n, b, work, work, -1); err != nil {
			panic(err)
		}
		lwkopt = max(6*n+16, 3*n+int(work.Get(0)))
		if err = Dormqr(Left, Trans, n, n, n, b, work, a, work, -1); err != nil {
			panic(err)
		}
		lwkopt = max(lwkopt, 3*n+int(work.Get(0)))
		if ilvsl {
			if err = Dorgqr(n, n, n, vsl, work, work, -1); err != nil {
				panic(err)
			}
			lwkopt = max(lwkopt, 3*n+int(work.Get(0)))
		}
		if err = Dgghd3(jobvsl, jobvsr, n, 1, n, a, b, vsl, vsr, work, -1); err != nil {
			panic(err)
		}
		lwkopt = max(lwkopt, 3*n+int(work.Get(0)))
		if ierr, err = Dhgeqz('S', jobvsl, jobvsr, n, 1, n, a, b, alphar, alphai, beta, vsl, vsr, work, -1); err != nil {
			panic(err)
		}
		lwkopt = max(lwkopt, 2*n+int(work.Get(0)))
		if wantst {
			if sdim, _, _, ierr, err = Dtgsen(0, ilvsl, ilvsr, *bwork, n, a, b, alphar, alphai, beta, vsl, vsr, dif, work, -1, &idum, 1); err != nil {
				panic(err)
			}
			lwkopt = max(lwkopt, 2*n+int(work.Get(0)))
		}
		work.Set(0, float64(lwkopt))
	}

	if err != nil {
		gltest.Xerbla2("Dgges3", err)
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
	safmin = Dlamch(SafeMinimum)
	safmax = one / safmin
	safmin, safmax = Dlabad(safmin, safmax)
	smlnum = math.Sqrt(safmin) / eps
	bignum = one / smlnum

	//     Scale A if max element outside range [SMLNUM,BIGNUM]
	anrm = Dlange('M', n, n, a, work)
	ilascl = false
	if anrm > zero && anrm < smlnum {
		anrmto = smlnum
		ilascl = true
	} else if anrm > bignum {
		anrmto = bignum
		ilascl = true
	}
	if ilascl {
		if err = Dlascl('G', 0, 0, anrm, anrmto, n, n, a); err != nil {
			panic(err)
		}
	}

	//     Scale B if max element outside range [SMLNUM,BIGNUM]
	bnrm = Dlange('M', n, n, b, work)
	ilbscl = false
	if bnrm > zero && bnrm < smlnum {
		bnrmto = smlnum
		ilbscl = true
	} else if bnrm > bignum {
		bnrmto = bignum
		ilbscl = true
	}
	if ilbscl {
		if err = Dlascl('G', 0, 0, bnrm, bnrmto, n, n, b); err != nil {
			panic(err)
		}
	}

	//     Permute the matrix to make it more nearly triangular
	ileft = 1
	iright = n + 1
	iwrk = iright + n
	if ilo, ihi, err = Dggbal('P', n, a, b, work.Off(ileft-1), work.Off(iright-1), work.Off(iwrk-1)); err != nil {
		panic(err)
	}

	//     Reduce B to triangular form (QR decomposition of B)
	irows = ihi + 1 - ilo
	icols = n + 1 - ilo
	itau = iwrk
	iwrk = itau + irows
	if err = Dgeqrf(irows, icols, b.Off(ilo-1, ilo-1), work.Off(itau-1), work.Off(iwrk-1), lwork+1-iwrk); err != nil {
		panic(err)
	}

	//     Apply the orthogonal transformation to matrix A
	if err = Dormqr(Left, Trans, irows, icols, irows, b.Off(ilo-1, ilo-1), work.Off(itau-1), a.Off(ilo-1, ilo-1), work.Off(iwrk-1), lwork+1-iwrk); err != nil {
		panic(err)
	}

	//     Initialize VSL
	if ilvsl {
		Dlaset(Full, n, n, zero, one, vsl)
		if irows > 1 {
			Dlacpy(Lower, irows-1, irows-1, b.Off(ilo, ilo-1), vsl.Off(ilo, ilo-1))
		}
		if err = Dorgqr(irows, irows, irows, vsl.Off(ilo-1, ilo-1), work.Off(itau-1), work.Off(iwrk-1), lwork+1-iwrk); err != nil {
			panic(err)
		}
	}

	//     Initialize VSR
	if ilvsr {
		Dlaset(Full, n, n, zero, one, vsr)
	}

	//     Reduce to generalized Hessenberg form
	if err = Dgghd3(jobvsl, jobvsr, n, ilo, ihi, a, b, vsl, vsr, work.Off(iwrk-1), lwork+1-iwrk); err != nil {
		panic(err)
	}

	//     Perform QZ algorithm, computing Schur vectors if desired
	iwrk = itau
	if ierr, err = Dhgeqz('S', jobvsl, jobvsr, n, ilo, ihi, a, b, alphar, alphai, beta, vsl, vsr, work.Off(iwrk-1), lwork+1-iwrk); err != nil {
		panic(err)
	}
	if ierr != 0 {
		if ierr > 0 && ierr <= n {
			info = ierr
		} else if ierr > n && ierr <= 2*n {
			info = ierr - n
		} else {
			info = n + 1
		}
		goto label50
	}

	//     Sort eigenvalues ALPHA/BETA if desired
	sdim = 0
	if wantst {
		//        Undo scaling on eigenvalues before SELCTGing
		if ilascl {
			if err = Dlascl('G', 0, 0, anrmto, anrm, n, 1, alphar.Matrix(n, opts)); err != nil {
				panic(err)
			}
			if err = Dlascl('G', 0, 0, anrmto, anrm, n, 1, alphai.Matrix(n, opts)); err != nil {
				panic(err)
			}
		}
		if ilbscl {
			if err = Dlascl('G', 0, 0, bnrmto, bnrm, n, 1, beta.Matrix(n, opts)); err != nil {
				panic(err)
			}
		}

		//        Select eigenvalues
		for i = 1; i <= n; i++ {
			(*bwork)[i-1] = selctg(alphar.GetPtr(i-1), alphai.GetPtr(i-1), beta.GetPtr(i-1))
		}

		if sdim, _, _, ierr, err = Dtgsen(0, ilvsl, ilvsr, *bwork, n, a, b, alphar, alphai, beta, vsl, vsr, dif, work.Off(iwrk-1), lwork-iwrk+1, &idum, 1); err != nil {
			panic(err)
		}
		if ierr == 1 {
			info = n + 3
		}

	}

	//     Apply back-permutation to VSL and VSR
	if ilvsl {
		if err = Dggbak('P', Left, n, ilo, ihi, work.Off(ileft-1), work.Off(iright-1), n, vsl); err != nil {
			panic(err)
		}
	}

	if ilvsr {
		if err = Dggbak('P', Right, n, ilo, ihi, work.Off(ileft-1), work.Off(iright-1), n, vsr); err != nil {
			panic(err)
		}
	}

	//     Check if unscaling would cause over/underflow, if so, rescale
	//     (ALPHAR(I),ALPHAI(I),BETA(I)) so BETA(I) is on the order of
	//     B(I,I) and ALPHAR(I) and ALPHAI(I) are on the order of A(I,I)
	if ilascl {
		for i = 1; i <= n; i++ {
			if alphai.Get(i-1) != zero {
				if (alphar.Get(i-1)/safmax) > (anrmto/anrm) || (safmin/alphar.Get(i-1)) > (anrm/anrmto) {
					work.Set(0, math.Abs(a.Get(i-1, i-1)/alphar.Get(i-1)))
					beta.Set(i-1, beta.Get(i-1)*work.Get(0))
					alphar.Set(i-1, alphar.Get(i-1)*work.Get(0))
					alphai.Set(i-1, alphai.Get(i-1)*work.Get(0))
				} else if (alphai.Get(i-1)/safmax) > (anrmto/anrm) || (safmin/alphai.Get(i-1)) > (anrm/anrmto) {
					work.Set(0, math.Abs(a.Get(i-1, i)/alphai.Get(i-1)))
					beta.Set(i-1, beta.Get(i-1)*work.Get(0))
					alphar.Set(i-1, alphar.Get(i-1)*work.Get(0))
					alphai.Set(i-1, alphai.Get(i-1)*work.Get(0))
				}
			}
		}
	}

	if ilbscl {
		for i = 1; i <= n; i++ {
			if alphai.Get(i-1) != zero {
				if (beta.Get(i-1)/safmax) > (bnrmto/bnrm) || (safmin/beta.Get(i-1)) > (bnrm/bnrmto) {
					work.Set(0, math.Abs(b.Get(i-1, i-1)/beta.Get(i-1)))
					beta.Set(i-1, beta.Get(i-1)*work.Get(0))
					alphar.Set(i-1, alphar.Get(i-1)*work.Get(0))
					alphai.Set(i-1, alphai.Get(i-1)*work.Get(0))
				}
			}
		}
	}

	//     Undo scaling
	if ilascl {
		if err = Dlascl('H', 0, 0, anrmto, anrm, n, n, a); err != nil {
			panic(err)
		}
		if err = Dlascl('G', 0, 0, anrmto, anrm, n, 1, alphar.Matrix(n, opts)); err != nil {
			panic(err)
		}
		if err = Dlascl('G', 0, 0, anrmto, anrm, n, 1, alphai.Matrix(n, opts)); err != nil {
			panic(err)
		}
	}

	if ilbscl {
		if err = Dlascl('U', 0, 0, bnrmto, bnrm, n, n, b); err != nil {
			panic(err)
		}
		if err = Dlascl('G', 0, 0, bnrmto, bnrm, n, 1, beta.Matrix(n, opts)); err != nil {
			panic(err)
		}
	}

	if wantst {
		//        Check if reordering is correct
		lastsl = true
		lst2sl = true
		sdim = 0
		ip = 0
		for i = 1; i <= n; i++ {
			cursl = selctg(alphar.GetPtr(i-1), alphai.GetPtr(i-1), beta.GetPtr(i-1))
			if alphai.Get(i-1) == zero {
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

label50:
	;

	work.Set(0, float64(lwkopt))

	return
}
