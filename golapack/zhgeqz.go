package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math/cmplx"
)

// Zhgeqz computes the eigenvalues of a complex matrix pair (H,T),
// where H is an upper Hessenberg matrix and T is upper triangular,
// using the single-shift QZ method.
// Matrix pairs of this _type are produced by the reduction to
// generalized upper Hessenberg form of a complex matrix pair (A,B):
//
//    A = Q1*H*Z1**H,  B = Q1*T*Z1**H,
//
// as computed by ZGGHRD.
//
// If JOB='S', then the Hessenberg-triangular pair (H,T) is
// also reduced to generalized Schur form,
//
//    H = Q*S*Z**H,  T = Q*P*Z**H,
//
// where Q and Z are unitary matrices and S and P are upper triangular.
//
// Optionally, the unitary matrix Q from the generalized Schur
// factorization may be postmultiplied into an input matrix Q1, and the
// unitary matrix Z may be postmultiplied into an input matrix Z1.
// If Q1 and Z1 are the unitary matrices from ZGGHRD that reduced
// the matrix pair (A,B) to generalized Hessenberg form, then the output
// matrices Q1*Q and Z1*Z are the unitary factors from the generalized
// Schur factorization of (A,B):
//
//    A = (Q1*Q)*S*(Z1*Z)**H,  B = (Q1*Q)*P*(Z1*Z)**H.
//
// To avoid overflow, eigenvalues of the matrix pair (H,T)
// (equivalently, of (A,B)) are computed as a pair of complex values
// (alpha,beta).  If beta is nonzero, lambda = alpha / beta is an
// eigenvalue of the generalized nonsymmetric eigenvalue problem (GNEP)
//    A*x = lambda*B*x
// and if alpha is nonzero, mu = beta / alpha is an eigenvalue of the
// alternate form of the GNEP
//    mu*A*y = B*y.
// The values of alpha and beta for the i-th eigenvalue can be read
// directly from the generalized Schur form:  alpha = S(i,i),
// beta = P(i,i).
//
// Ref: C.B. Moler & G.W. Stewart, "An Algorithm for Generalized Matrix
//      Eigenvalue Problems", SIAM J. Numer. Anal., 10(1973),
//      pp. 241--256.
func Zhgeqz(job, compq, compz byte, n, ilo, ihi *int, h *mat.CMatrix, ldh *int, t *mat.CMatrix, ldt *int, alpha, beta *mat.CVector, q *mat.CMatrix, ldq *int, z *mat.CMatrix, ldz *int, work *mat.CVector, lwork *int, rwork *mat.Vector, info *int) {
	var ilazr2, ilazro, ilq, ilschr, ilz, lquery bool
	var abi22, ad11, ad12, ad21, ad22, cone, ctemp, ctemp2, ctemp3, czero, eshift, rtdisc, s, shift, signbc, t1, u12 complex128
	var absb, anorm, ascale, atol, bnorm, bscale, btol, c, half, one, safmin, temp, temp2, tempr, ulp, zero float64
	var icompq, icompz, ifirst, ifrstm, iiter, ilast, ilastm, in, ischur, istart, j, jc, jch, jiter, jr, maxit int

	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	zero = 0.0
	one = 1.0
	half = 0.5

	//     Decode JOB, COMPQ, COMPZ
	if job == 'E' {
		ilschr = false
		ischur = 1
	} else if job == 'S' {
		ilschr = true
		ischur = 2
	} else {
		ischur = 0
	}

	if compq == 'N' {
		ilq = false
		icompq = 1
	} else if compq == 'V' {
		ilq = true
		icompq = 2
	} else if compq == 'I' {
		ilq = true
		icompq = 3
	} else {
		icompq = 0
	}

	if compz == 'N' {
		ilz = false
		icompz = 1
	} else if compz == 'V' {
		ilz = true
		icompz = 2
	} else if compz == 'I' {
		ilz = true
		icompz = 3
	} else {
		icompz = 0
	}

	//     Check Argument Values
	(*info) = 0
	work.SetRe(0, float64(maxint(1, *n)))
	lquery = ((*lwork) == -1)
	if ischur == 0 {
		(*info) = -1
	} else if icompq == 0 {
		(*info) = -2
	} else if icompz == 0 {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*ilo) < 1 {
		(*info) = -5
	} else if (*ihi) > (*n) || (*ihi) < (*ilo)-1 {
		(*info) = -6
	} else if (*ldh) < (*n) {
		(*info) = -8
	} else if (*ldt) < (*n) {
		(*info) = -10
	} else if (*ldq) < 1 || (ilq && (*ldq) < (*n)) {
		(*info) = -14
	} else if (*ldz) < 1 || (ilz && (*ldz) < (*n)) {
		(*info) = -16
	} else if (*lwork) < maxint(1, *n) && !lquery {
		(*info) = -18
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHGEQZ"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	//
	//     WORK( 1 ) = CMPLX( 1 )
	if (*n) <= 0 {
		work.SetRe(0, 1)
		return
	}

	//     Initialize Q and Z
	if icompq == 3 {
		Zlaset('F', n, n, &czero, &cone, q, ldq)
	}
	if icompz == 3 {
		Zlaset('F', n, n, &czero, &cone, z, ldz)
	}

	//     Machine Constants
	in = (*ihi) + 1 - (*ilo)
	safmin = Dlamch(SafeMinimum)
	ulp = Dlamch(Epsilon) * Dlamch(Base)
	anorm = Zlanhs('F', &in, h.Off((*ilo)-1, (*ilo)-1), ldh, rwork)
	bnorm = Zlanhs('F', &in, t.Off((*ilo)-1, (*ilo)-1), ldt, rwork)
	atol = maxf64(safmin, ulp*anorm)
	btol = maxf64(safmin, ulp*bnorm)
	ascale = one / maxf64(safmin, anorm)
	bscale = one / maxf64(safmin, bnorm)

	//     Set Eigenvalues IHI+1:N
	for j = (*ihi) + 1; j <= (*n); j++ {
		absb = t.GetMag(j-1, j-1)
		if absb > safmin {
			signbc = cmplx.Conj(t.Get(j-1, j-1) / complex(absb, 0))
			t.SetRe(j-1, j-1, absb)
			if ilschr {
				goblas.Zscal(toPtr(j-1), &signbc, t.CVector(0, j-1), func() *int { y := 1; return &y }())
				goblas.Zscal(&j, &signbc, h.CVector(0, j-1), func() *int { y := 1; return &y }())
			} else {
				goblas.Zscal(func() *int { y := 1; return &y }(), &signbc, h.CVector(j-1, j-1), func() *int { y := 1; return &y }())
			}
			if ilz {
				goblas.Zscal(n, &signbc, z.CVector(0, j-1), func() *int { y := 1; return &y }())
			}
		} else {
			t.Set(j-1, j-1, czero)
		}
		alpha.Set(j-1, h.Get(j-1, j-1))
		beta.Set(j-1, t.Get(j-1, j-1))
	}

	//     If IHI < ILO, skip QZ steps
	if (*ihi) < (*ilo) {
		goto label190
	}

	//     MAIN QZ ITERATION LOOP
	//
	//     Initialize dynamic indices
	//
	//     Eigenvalues ILAST+1:N have been found.
	//        Column operations modify rows IFRSTM:whatever
	//        Row operations modify columns whatever:ILASTM
	//
	//     If only eigenvalues are being computed, then
	//        IFRSTM is the row of the last splitting row above row ILAST;
	//        this is always at least ILO.
	//     IITER counts iterations since the last eigenvalue was found,
	//        to tell when to use an extraordinary shift.
	//     MAXIT is the maximum number of QZ sweeps allowed.
	ilast = (*ihi)
	if ilschr {
		ifrstm = 1
		ilastm = (*n)
	} else {
		ifrstm = (*ilo)
		ilastm = (*ihi)
	}
	iiter = 0
	eshift = czero
	maxit = 30 * ((*ihi) - (*ilo) + 1)

	for jiter = 1; jiter <= maxit; jiter++ {
		//        Check for too many iterations.
		if jiter > maxit {
			goto label180
		}

		//        Split the matrix if possible.
		//
		//        Two tests:
		//           1: H(j,j-1)=0  or  j=ILO
		//           2: T(j,j)=0
		//
		//        Special case: j=ILAST
		if ilast == (*ilo) {
			goto label60
		} else {
			if abs1(h.Get(ilast-1, ilast-1-1)) <= atol {
				h.Set(ilast-1, ilast-1-1, czero)
				goto label60
			}
		}

		if t.GetMag(ilast-1, ilast-1) <= btol {
			t.Set(ilast-1, ilast-1, czero)
			goto label50
		}

		//        General case: j<ILAST
		for j = ilast - 1; j >= (*ilo); j-- {
			//           Test 1: for H(j,j-1)=0 or j=ILO
			if j == (*ilo) {
				ilazro = true
			} else {
				if abs1(h.Get(j-1, j-1-1)) <= atol {
					h.Set(j-1, j-1-1, czero)
					ilazro = true
				} else {
					ilazro = false
				}
			}

			//           Test 2: for T(j,j)=0
			if t.GetMag(j-1, j-1) < btol {
				t.Set(j-1, j-1, czero)

				//              Test 1a: Check for 2 consecutive small subdiagonals in A
				ilazr2 = false
				if !ilazro {
					if abs1(h.Get(j-1, j-1-1))*(ascale*abs1(h.Get(j+1-1, j-1))) <= abs1(h.Get(j-1, j-1))*(ascale*atol) {
						ilazr2 = true
					}
				}

				//              If both tests pass (1 & 2), i.e., the leading diagonal
				//              element of B in the block is zero, split a 1x1 block off
				//              at the top. (I.e., at the J-th row/column) The leading
				//              diagonal element of the remainder can also be zero, so
				//              this may have to be done repeatedly.
				if ilazro || ilazr2 {
					for jch = j; jch <= ilast-1; jch++ {
						ctemp = h.Get(jch-1, jch-1)
						Zlartg(&ctemp, h.GetPtr(jch+1-1, jch-1), &c, &s, h.GetPtr(jch-1, jch-1))
						h.Set(jch+1-1, jch-1, czero)
						Zrot(toPtr(ilastm-jch), h.CVector(jch-1, jch+1-1), ldh, h.CVector(jch+1-1, jch+1-1), ldh, &c, &s)
						Zrot(toPtr(ilastm-jch), t.CVector(jch-1, jch+1-1), ldt, t.CVector(jch+1-1, jch+1-1), ldt, &c, &s)
						if ilq {
							Zrot(n, q.CVector(0, jch-1), func() *int { y := 1; return &y }(), q.CVector(0, jch+1-1), func() *int { y := 1; return &y }(), &c, toPtrc128(cmplx.Conj(s)))
						}
						if ilazr2 {
							h.Set(jch-1, jch-1-1, h.Get(jch-1, jch-1-1)*complex(c, 0))
						}
						ilazr2 = false
						if abs1(t.Get(jch+1-1, jch+1-1)) >= btol {
							if jch+1 >= ilast {
								goto label60
							} else {
								ifirst = jch + 1
								goto label70
							}
						}
						t.Set(jch+1-1, jch+1-1, czero)
					}
					goto label50
				} else {
					//                 Only test 2 passed -- chase the zero to T(ILAST,ILAST)
					//                 Then process as in the case T(ILAST,ILAST)=0
					for jch = j; jch <= ilast-1; jch++ {
						ctemp = t.Get(jch-1, jch+1-1)
						Zlartg(&ctemp, t.GetPtr(jch+1-1, jch+1-1), &c, &s, t.GetPtr(jch-1, jch+1-1))
						t.Set(jch+1-1, jch+1-1, czero)
						if jch < ilastm-1 {
							Zrot(toPtr(ilastm-jch-1), t.CVector(jch-1, jch+2-1), ldt, t.CVector(jch+1-1, jch+2-1), ldt, &c, &s)
						}
						Zrot(toPtr(ilastm-jch+2), h.CVector(jch-1, jch-1-1), ldh, h.CVector(jch+1-1, jch-1-1), ldh, &c, &s)
						if ilq {
							Zrot(n, q.CVector(0, jch-1), func() *int { y := 1; return &y }(), q.CVector(0, jch+1-1), func() *int { y := 1; return &y }(), &c, toPtrc128(cmplx.Conj(s)))
						}
						ctemp = h.Get(jch+1-1, jch-1)
						Zlartg(&ctemp, h.GetPtr(jch+1-1, jch-1-1), &c, &s, h.GetPtr(jch+1-1, jch-1))
						h.Set(jch+1-1, jch-1-1, czero)
						Zrot(toPtr(jch+1-ifrstm), h.CVector(ifrstm-1, jch-1), func() *int { y := 1; return &y }(), h.CVector(ifrstm-1, jch-1-1), func() *int { y := 1; return &y }(), &c, &s)
						Zrot(toPtr(jch-ifrstm), t.CVector(ifrstm-1, jch-1), func() *int { y := 1; return &y }(), t.CVector(ifrstm-1, jch-1-1), func() *int { y := 1; return &y }(), &c, &s)
						if ilz {
							Zrot(n, z.CVector(0, jch-1), func() *int { y := 1; return &y }(), z.CVector(0, jch-1-1), func() *int { y := 1; return &y }(), &c, &s)
						}
					}
					goto label50
				}
			} else if ilazro {
				//              Only test 1 passed -- work on J:ILAST
				ifirst = j
				goto label70
			}

			//           Neither test passed -- try next J
		}

		//        (Drop-through is "impossible")
		(*info) = 2*(*n) + 1
		goto label210

		//        T(ILAST,ILAST)=0 -- clear H(ILAST,ILAST-1) to split off a
		//        1x1 block.
	label50:
		;
		ctemp = h.Get(ilast-1, ilast-1)
		Zlartg(&ctemp, h.GetPtr(ilast-1, ilast-1-1), &c, &s, h.GetPtr(ilast-1, ilast-1))
		h.Set(ilast-1, ilast-1-1, czero)
		Zrot(toPtr(ilast-ifrstm), h.CVector(ifrstm-1, ilast-1), func() *int { y := 1; return &y }(), h.CVector(ifrstm-1, ilast-1-1), func() *int { y := 1; return &y }(), &c, &s)
		Zrot(toPtr(ilast-ifrstm), t.CVector(ifrstm-1, ilast-1), func() *int { y := 1; return &y }(), t.CVector(ifrstm-1, ilast-1-1), func() *int { y := 1; return &y }(), &c, &s)
		if ilz {
			Zrot(n, z.CVector(0, ilast-1), func() *int { y := 1; return &y }(), z.CVector(0, ilast-1-1), func() *int { y := 1; return &y }(), &c, &s)
		}

		//        H(ILAST,ILAST-1)=0 -- Standardize B, set ALPHA and BETA
	label60:
		;
		absb = t.GetMag(ilast-1, ilast-1)
		if absb > safmin {
			signbc = cmplx.Conj(t.Get(ilast-1, ilast-1) / complex(absb, 0))
			t.SetRe(ilast-1, ilast-1, absb)
			if ilschr {
				goblas.Zscal(toPtr(ilast-ifrstm), &signbc, t.CVector(ifrstm-1, ilast-1), func() *int { y := 1; return &y }())
				goblas.Zscal(toPtr(ilast+1-ifrstm), &signbc, h.CVector(ifrstm-1, ilast-1), func() *int { y := 1; return &y }())
			} else {
				goblas.Zscal(func() *int { y := 1; return &y }(), &signbc, h.CVector(ilast-1, ilast-1), func() *int { y := 1; return &y }())
			}
			if ilz {
				goblas.Zscal(n, &signbc, z.CVector(0, ilast-1), func() *int { y := 1; return &y }())
			}
		} else {
			t.Set(ilast-1, ilast-1, czero)
		}
		alpha.Set(ilast-1, h.Get(ilast-1, ilast-1))
		beta.Set(ilast-1, t.Get(ilast-1, ilast-1))

		//        Go to next block -- exit if finished.
		ilast = ilast - 1
		if ilast < (*ilo) {
			goto label190
		}

		//        Reset counters
		iiter = 0
		eshift = czero
		if !ilschr {
			ilastm = ilast
			if ifrstm > ilast {
				ifrstm = (*ilo)
			}
		}
		goto label160

		//        QZ step
		//
		//        This iteration only involves rows/columns IFIRST:ILAST.  We
		//        assume IFIRST < ILAST, and that the diagonal of B is non-zero.
	label70:
		;
		iiter = iiter + 1
		if !ilschr {
			ifrstm = ifirst
		}

		//        Compute the Shift.
		//
		//        At this point, IFIRST < ILAST, and the diagonal elements of
		//        T(IFIRST:ILAST,IFIRST,ILAST) are larger than BTOL (in
		//        magnitude)
		if (iiter/10)*10 != iiter {
			//           The Wilkinson shift (AEP p.512), i.e., the eigenvalue of
			//           the bottom-right 2x2 block of A inv(B) which is nearest to
			//           the bottom-right element.
			//
			//           We factor B as U*D, where U has unit diagonals, and
			//           compute (A*inv(D))*inv(U).
			u12 = (complex(bscale, 0) * t.Get(ilast-1-1, ilast-1)) / (complex(bscale, 0) * t.Get(ilast-1, ilast-1))
			ad11 = (complex(ascale, 0) * h.Get(ilast-1-1, ilast-1-1)) / (complex(bscale, 0) * t.Get(ilast-1-1, ilast-1-1))
			ad21 = (complex(ascale, 0) * h.Get(ilast-1, ilast-1-1)) / (complex(bscale, 0) * t.Get(ilast-1-1, ilast-1-1))
			ad12 = (complex(ascale, 0) * h.Get(ilast-1-1, ilast-1)) / (complex(bscale, 0) * t.Get(ilast-1, ilast-1))
			ad22 = (complex(ascale, 0) * h.Get(ilast-1, ilast-1)) / (complex(bscale, 0) * t.Get(ilast-1, ilast-1))
			abi22 = ad22 - u12*ad21
			//
			t1 = complex(half, 0) * (ad11 + abi22)
			rtdisc = cmplx.Sqrt(cmplx.Pow(t1, 2) + ad12*ad21 - ad11*ad22)
			temp = real(t1-abi22)*real(rtdisc) + imag(t1-abi22)*imag(rtdisc)
			if temp <= zero {
				shift = t1 + rtdisc
			} else {
				shift = t1 - rtdisc
			}
		} else {
			//           Exceptional shift.  Chosen for no particularly good reason.
			eshift = eshift + (complex(ascale, 0)*h.Get(ilast-1, ilast-1-1))/(complex(bscale, 0)*t.Get(ilast-1-1, ilast-1-1))
			shift = eshift
		}

		//        Now check for two consecutive small subdiagonals.
		for j = ilast - 1; j >= ifirst+1; j-- {
			istart = j
			ctemp = complex(ascale, 0)*h.Get(j-1, j-1) - shift*(complex(bscale, 0)*t.Get(j-1, j-1))
			temp = abs1(ctemp)
			temp2 = ascale * abs1(h.Get(j+1-1, j-1))
			tempr = maxf64(temp, temp2)
			if tempr < one && tempr != zero {
				temp = temp / tempr
				temp2 = temp2 / tempr
			}
			if abs1(h.Get(j-1, j-1-1))*temp2 <= temp*atol {
				goto label90
			}
		}

		istart = ifirst
		ctemp = complex(ascale, 0)*h.Get(ifirst-1, ifirst-1) - shift*(complex(bscale, 0)*t.Get(ifirst-1, ifirst-1))
	label90:
		;

		//        Do an implicit-shift QZ sweep.
		//
		//        Initial Q
		ctemp2 = complex(ascale, 0) * h.Get(istart+1-1, istart-1)
		Zlartg(&ctemp, &ctemp2, &c, &s, &ctemp3)

		//        Sweep
		for j = istart; j <= ilast-1; j++ {
			if j > istart {
				ctemp = h.Get(j-1, j-1-1)
				Zlartg(&ctemp, h.GetPtr(j+1-1, j-1-1), &c, &s, h.GetPtr(j-1, j-1-1))
				h.Set(j+1-1, j-1-1, czero)
			}

			for jc = j; jc <= ilastm; jc++ {
				ctemp = complex(c, 0)*h.Get(j-1, jc-1) + s*h.Get(j+1-1, jc-1)
				h.Set(j+1-1, jc-1, -cmplx.Conj(s)*h.Get(j-1, jc-1)+complex(c, 0)*h.Get(j+1-1, jc-1))
				h.Set(j-1, jc-1, ctemp)
				ctemp2 = complex(c, 0)*t.Get(j-1, jc-1) + s*t.Get(j+1-1, jc-1)
				t.Set(j+1-1, jc-1, -cmplx.Conj(s)*t.Get(j-1, jc-1)+complex(c, 0)*t.Get(j+1-1, jc-1))
				t.Set(j-1, jc-1, ctemp2)
			}
			if ilq {
				for jr = 1; jr <= (*n); jr++ {
					ctemp = complex(c, 0)*q.Get(jr-1, j-1) + cmplx.Conj(s)*q.Get(jr-1, j+1-1)
					q.Set(jr-1, j+1-1, -s*q.Get(jr-1, j-1)+complex(c, 0)*q.Get(jr-1, j+1-1))
					q.Set(jr-1, j-1, ctemp)
				}
			}

			ctemp = t.Get(j+1-1, j+1-1)
			Zlartg(&ctemp, t.GetPtr(j+1-1, j-1), &c, &s, t.GetPtr(j+1-1, j+1-1))
			t.Set(j+1-1, j-1, czero)

			for jr = ifrstm; jr <= minint(j+2, ilast); jr++ {
				ctemp = complex(c, 0)*h.Get(jr-1, j+1-1) + s*h.Get(jr-1, j-1)
				h.Set(jr-1, j-1, -cmplx.Conj(s)*h.Get(jr-1, j+1-1)+complex(c, 0)*h.Get(jr-1, j-1))
				h.Set(jr-1, j+1-1, ctemp)
			}
			for jr = ifrstm; jr <= j; jr++ {
				ctemp = complex(c, 0)*t.Get(jr-1, j+1-1) + s*t.Get(jr-1, j-1)
				t.Set(jr-1, j-1, -cmplx.Conj(s)*t.Get(jr-1, j+1-1)+complex(c, 0)*t.Get(jr-1, j-1))
				t.Set(jr-1, j+1-1, ctemp)
			}
			if ilz {
				for jr = 1; jr <= (*n); jr++ {
					ctemp = complex(c, 0)*z.Get(jr-1, j+1-1) + s*z.Get(jr-1, j-1)
					z.Set(jr-1, j-1, -cmplx.Conj(s)*z.Get(jr-1, j+1-1)+complex(c, 0)*z.Get(jr-1, j-1))
					z.Set(jr-1, j+1-1, ctemp)
				}
			}
		}

	label160:
	}

	//     Drop-through = non-convergence
label180:
	;
	(*info) = ilast
	goto label210

	//     Successful completion of all QZ steps
label190:
	;

	//     Set Eigenvalues 1:ILO-1
	for j = 1; j <= (*ilo)-1; j++ {
		absb = t.GetMag(j-1, j-1)
		if absb > safmin {
			signbc = cmplx.Conj(t.Get(j-1, j-1) / complex(absb, 0))
			t.SetRe(j-1, j-1, absb)
			if ilschr {
				goblas.Zscal(toPtr(j-1), &signbc, t.CVector(0, j-1), func() *int { y := 1; return &y }())
				goblas.Zscal(&j, &signbc, h.CVector(0, j-1), func() *int { y := 1; return &y }())
			} else {
				goblas.Zscal(func() *int { y := 1; return &y }(), &signbc, h.CVector(j-1, j-1), func() *int { y := 1; return &y }())
			}
			if ilz {
				goblas.Zscal(n, &signbc, z.CVector(0, j-1), func() *int { y := 1; return &y }())
			}
		} else {
			t.Set(j-1, j-1, czero)
		}
		alpha.Set(j-1, h.Get(j-1, j-1))
		beta.Set(j-1, t.Get(j-1, j-1))
	}

	//     Normal Termination
	(*info) = 0

	//     Exit (other than argument error) -- return optimal workspace size
label210:
	;
	work.SetRe(0, float64(*n))
}
