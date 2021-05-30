package golapack

import (
	"golinalg/goblas"
	"golinalg/mat"
)

// Ztprfb applies a complex "triangular-pentagonal" block reflector H or its
// conjugate transpose H**H to a complex matrix C, which is composed of two
// blocks A and B, either from the left or right.
func Ztprfb(side, trans, direct, storev byte, m, n, k, l *int, v *mat.CMatrix, ldv *int, t *mat.CMatrix, ldt *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, work *mat.CMatrix, ldwork *int) {
	var backward, column, forward, left, right, row bool
	var one, zero complex128
	var i, j, kp, mp, np int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Quick return if possible
	if (*m) <= 0 || (*n) <= 0 || (*k) <= 0 || (*l) < 0 {
		return
	}

	if storev == 'C' {
		column = true
		row = false
	} else if storev == 'R' {
		column = false
		row = true
	} else {
		column = false
		row = false
	}

	if side == 'L' {
		left = true
		right = false
	} else if side == 'R' {
		left = false
		right = true
	} else {
		left = false
		right = false
	}

	if direct == 'F' {
		forward = true
		backward = false
	} else if direct == 'B' {
		forward = false
		backward = true
	} else {
		forward = false
		backward = false
	}

	// ---------------------------------------------------------------------------
	if column && forward && left {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ I ]    (K-by-K)
		//                  [ V ]    (M-by-K)
		//
		//        Form  H C  or  H**H C  where  C = [ A ]  (K-by-N)
		//                                          [ B ]  (M-by-N)
		//
		//        H = I - W T W**H          or  H**H = I - W T**H W**H
		//
		//        A = A -   T (A + V**H B)  or  A = A -   T**H (A + V**H B)
		//        B = B - V T (A + V**H B)  or  B = B - V T**H (A + V**H B)
		//
		// ---------------------------------------------------------------------------
		mp = minint((*m)-(*l)+1, *m)
		kp = minint((*l)+1, *k)

		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*l); i++ {
				work.Set(i-1, j-1, b.Get((*m)-(*l)+i-1, j-1))
			}
		}
		goblas.Ztrmm(Left, Upper, ConjTrans, NonUnit, l, n, &one, v.Off(mp-1, 0), ldv, work, ldwork)
		goblas.Zgemm(ConjTrans, NoTrans, l, n, toPtr((*m)-(*l)), &one, v, ldv, b, ldb, &one, work, ldwork)
		goblas.Zgemm(ConjTrans, NoTrans, toPtr((*k)-(*l)), n, m, &one, v.Off(0, kp-1), ldv, b, ldb, &zero, work.Off(kp-1, 0), ldwork)

		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*k); i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		goblas.Ztrmm(Left, Upper, mat.TransByte(trans), NonUnit, k, n, &one, t, ldt, work, ldwork)

		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*k); i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		goblas.Zgemm(NoTrans, NoTrans, toPtr((*m)-(*l)), n, k, toPtrc128(-one), v, ldv, work, ldwork, &one, b, ldb)
		goblas.Zgemm(NoTrans, NoTrans, l, n, toPtr((*k)-(*l)), toPtrc128(-one), v.Off(mp-1, kp-1), ldv, work.Off(kp-1, 0), ldwork, &one, b.Off(mp-1, 0), ldb)
		goblas.Ztrmm(Left, Upper, NoTrans, NonUnit, l, n, &one, v.Off(mp-1, 0), ldv, work, ldwork)
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*l); i++ {
				b.Set((*m)-(*l)+i-1, j-1, b.Get((*m)-(*l)+i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		// ---------------------------------------------------------------------------
	} else if column && forward && right {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ I ]    (K-by-K)
		//                  [ V ]    (N-by-K)
		//
		//        Form  C H or  C H**H  where  C = [ A B ] (A is M-by-K, B is M-by-N)
		//
		//        H = I - W T W**H          or  H**H = I - W T**H W**H
		//
		//        A = A - (A + B V) T      or  A = A - (A + B V) T**H
		//        B = B - (A + B V) T V**H  or  B = B - (A + B V) T**H V**H
		//
		// ---------------------------------------------------------------------------
		np = minint((*n)-(*l)+1, *n)
		kp = minint((*l)+1, *k)

		for j = 1; j <= (*l); j++ {
			for i = 1; i <= (*m); i++ {
				work.Set(i-1, j-1, b.Get(i-1, (*n)-(*l)+j-1))
			}
		}
		goblas.Ztrmm(Right, Upper, NoTrans, NonUnit, m, l, &one, v.Off(np-1, 0), ldv, work, ldwork)
		goblas.Zgemm(NoTrans, NoTrans, m, l, toPtr((*n)-(*l)), &one, b, ldb, v, ldv, &one, work, ldwork)
		goblas.Zgemm(NoTrans, NoTrans, m, toPtr((*k)-(*l)), n, &one, b, ldb, v.Off(0, kp-1), ldv, &zero, work.Off(0, kp-1), ldwork)

		for j = 1; j <= (*k); j++ {
			for i = 1; i <= (*m); i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		goblas.Ztrmm(Right, Upper, mat.TransByte(trans), NonUnit, m, k, &one, t, ldt, work, ldwork)

		for j = 1; j <= (*k); j++ {
			for i = 1; i <= (*m); i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		goblas.Zgemm(NoTrans, ConjTrans, m, toPtr((*n)-(*l)), k, toPtrc128(-one), work, ldwork, v, ldv, &one, b, ldb)
		goblas.Zgemm(NoTrans, ConjTrans, m, l, toPtr((*k)-(*l)), toPtrc128(-one), work.Off(0, kp-1), ldwork, v.Off(np-1, kp-1), ldv, &one, b.Off(0, np-1), ldb)
		goblas.Ztrmm(Right, Upper, ConjTrans, NonUnit, m, l, &one, v.Off(np-1, 0), ldv, work, ldwork)
		for j = 1; j <= (*l); j++ {
			for i = 1; i <= (*m); i++ {
				b.Set(i-1, (*n)-(*l)+j-1, b.Get(i-1, (*n)-(*l)+j-1)-work.Get(i-1, j-1))
			}
		}

		// ---------------------------------------------------------------------------
	} else if column && backward && left {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ V ]    (M-by-K)
		//                  [ I ]    (K-by-K)
		//
		//        Form  H C  or  H**H C  where  C = [ B ]  (M-by-N)
		//                                          [ A ]  (K-by-N)
		//
		//        H = I - W T W**H          or  H**H = I - W T**H W**H
		//
		//        A = A -   T (A + V**H B)  or  A = A -   T**H (A + V**H B)
		//        B = B - V T (A + V**H B)  or  B = B - V T**H (A + V**H B)
		//
		// ---------------------------------------------------------------------------
		mp = minint((*l)+1, *m)
		kp = minint((*k)-(*l)+1, *k)

		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*l); i++ {
				work.Set((*k)-(*l)+i-1, j-1, b.Get(i-1, j-1))
			}
		}

		goblas.Ztrmm(Left, Lower, ConjTrans, NonUnit, l, n, &one, v.Off(0, kp-1), ldv, work.Off(kp-1, 0), ldwork)
		goblas.Zgemm(ConjTrans, NoTrans, l, n, toPtr((*m)-(*l)), &one, v.Off(mp-1, kp-1), ldv, b.Off(mp-1, 0), ldb, &one, work.Off(kp-1, 0), ldwork)
		goblas.Zgemm(ConjTrans, NoTrans, toPtr((*k)-(*l)), n, m, &one, v, ldv, b, ldb, &zero, work, ldwork)

		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*k); i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		goblas.Ztrmm(Left, Lower, mat.TransByte(trans), NonUnit, k, n, &one, t, ldt, work, ldwork)

		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*k); i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		goblas.Zgemm(NoTrans, NoTrans, toPtr((*m)-(*l)), n, k, toPtrc128(-one), v.Off(mp-1, 0), ldv, work, ldwork, &one, b.Off(mp-1, 0), ldb)
		goblas.Zgemm(NoTrans, NoTrans, l, n, toPtr((*k)-(*l)), toPtrc128(-one), v, ldv, work, ldwork, &one, b, ldb)
		goblas.Ztrmm(Left, Lower, NoTrans, NonUnit, l, n, &one, v.Off(0, kp-1), ldv, work.Off(kp-1, 0), ldwork)
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*l); i++ {
				b.Set(i-1, j-1, b.Get(i-1, j-1)-work.Get((*k)-(*l)+i-1, j-1))
			}
		}

		// ---------------------------------------------------------------------------
	} else if column && backward && right {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ V ]    (N-by-K)
		//                  [ I ]    (K-by-K)
		//
		//        Form  C H  or  C H**H  where  C = [ B A ] (B is M-by-N, A is M-by-K)
		//
		//        H = I - W T W**H          or  H**H = I - W T**H W**H
		//
		//        A = A - (A + B V) T      or  A = A - (A + B V) T**H
		//        B = B - (A + B V) T V**H  or  B = B - (A + B V) T**H V**H
		//
		// ---------------------------------------------------------------------------
		np = minint((*l)+1, *n)
		kp = minint((*k)-(*l)+1, *k)

		for j = 1; j <= (*l); j++ {
			for i = 1; i <= (*m); i++ {
				work.Set(i-1, (*k)-(*l)+j-1, b.Get(i-1, j-1))
			}
		}
		goblas.Ztrmm(Right, Lower, NoTrans, NonUnit, m, l, &one, v.Off(0, kp-1), ldv, work.Off(0, kp-1), ldwork)
		goblas.Zgemm(NoTrans, NoTrans, m, l, toPtr((*n)-(*l)), &one, b.Off(0, np-1), ldb, v.Off(np-1, kp-1), ldv, &one, work.Off(0, kp-1), ldwork)
		goblas.Zgemm(NoTrans, NoTrans, m, toPtr((*k)-(*l)), n, &one, b, ldb, v, ldv, &zero, work, ldwork)

		for j = 1; j <= (*k); j++ {
			for i = 1; i <= (*m); i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		goblas.Ztrmm(Right, Lower, mat.TransByte(trans), NonUnit, m, k, &one, t, ldt, work, ldwork)

		for j = 1; j <= (*k); j++ {
			for i = 1; i <= (*m); i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		goblas.Zgemm(NoTrans, ConjTrans, m, toPtr((*n)-(*l)), k, toPtrc128(-one), work, ldwork, v.Off(np-1, 0), ldv, &one, b.Off(0, np-1), ldb)
		goblas.Zgemm(NoTrans, ConjTrans, m, l, toPtr((*k)-(*l)), toPtrc128(-one), work, ldwork, v, ldv, &one, b, ldb)
		goblas.Ztrmm(Right, Lower, ConjTrans, NonUnit, m, l, &one, v.Off(0, kp-1), ldv, work.Off(0, kp-1), ldwork)
		for j = 1; j <= (*l); j++ {
			for i = 1; i <= (*m); i++ {
				b.Set(i-1, j-1, b.Get(i-1, j-1)-work.Get(i-1, (*k)-(*l)+j-1))
			}
		}

		// ---------------------------------------------------------------------------
	} else if row && forward && left {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ I V ] ( I is K-by-K, V is K-by-M )
		//
		//        Form  H C  or  H**H C  where  C = [ A ]  (K-by-N)
		//                                          [ B ]  (M-by-N)
		//
		//        H = I - W**H T W          or  H**H = I - W**H T**H W
		//
		//        A = A -     T (A + V B)  or  A = A -     T**H (A + V B)
		//        B = B - V**H T (A + V B)  or  B = B - V**H T**H (A + V B)
		//
		// ---------------------------------------------------------------------------
		mp = minint((*m)-(*l)+1, *m)
		kp = minint((*l)+1, *k)

		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*l); i++ {
				work.Set(i-1, j-1, b.Get((*m)-(*l)+i-1, j-1))
			}
		}
		goblas.Ztrmm(Left, Lower, NoTrans, NonUnit, l, n, &one, v.Off(0, mp-1), ldv, work, ldb)
		goblas.Zgemm(NoTrans, NoTrans, l, n, toPtr((*m)-(*l)), &one, v, ldv, b, ldb, &one, work, ldwork)
		goblas.Zgemm(NoTrans, NoTrans, toPtr((*k)-(*l)), n, m, &one, v.Off(kp-1, 0), ldv, b, ldb, &zero, work.Off(kp-1, 0), ldwork)

		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*k); i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		goblas.Ztrmm(Left, Upper, mat.TransByte(trans), NonUnit, k, n, &one, t, ldt, work, ldwork)
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*k); i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		goblas.Zgemm(ConjTrans, NoTrans, toPtr((*m)-(*l)), n, k, toPtrc128(-one), v, ldv, work, ldwork, &one, b, ldb)
		goblas.Zgemm(ConjTrans, NoTrans, l, n, toPtr((*k)-(*l)), toPtrc128(-one), v.Off(kp-1, mp-1), ldv, work.Off(kp-1, 0), ldwork, &one, b.Off(mp-1, 0), ldb)
		goblas.Ztrmm(Left, Lower, ConjTrans, NonUnit, l, n, &one, v.Off(0, mp-1), ldv, work, ldwork)
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*l); i++ {
				b.Set((*m)-(*l)+i-1, j-1, b.Get((*m)-(*l)+i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		// ---------------------------------------------------------------------------
	} else if row && forward && right {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ I V ] ( I is K-by-K, V is K-by-N )
		//
		//        Form  C H  or  C H**H  where  C = [ A B ] (A is M-by-K, B is M-by-N)
		//
		//        H = I - W**H T W            or  H**H = I - W**H T**H W
		//
		//        A = A - (A + B V**H) T      or  A = A - (A + B V**H) T**H
		//        B = B - (A + B V**H) T V    or  B = B - (A + B V**H) T**H V
		//
		// ---------------------------------------------------------------------------
		np = minint((*n)-(*l)+1, *n)
		kp = minint((*l)+1, *k)

		for j = 1; j <= (*l); j++ {
			for i = 1; i <= (*m); i++ {
				work.Set(i-1, j-1, b.Get(i-1, (*n)-(*l)+j-1))
			}
		}
		goblas.Ztrmm(Right, Lower, ConjTrans, NonUnit, m, l, &one, v.Off(0, np-1), ldv, work, ldwork)
		goblas.Zgemm(NoTrans, ConjTrans, m, l, toPtr((*n)-(*l)), &one, b, ldb, v, ldv, &one, work, ldwork)
		goblas.Zgemm(NoTrans, ConjTrans, m, toPtr((*k)-(*l)), n, &one, b, ldb, v.Off(kp-1, 0), ldv, &zero, work.Off(0, kp-1), ldwork)

		for j = 1; j <= (*k); j++ {
			for i = 1; i <= (*m); i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		goblas.Ztrmm(Right, Upper, mat.TransByte(trans), NonUnit, m, k, &one, t, ldt, work, ldwork)

		for j = 1; j <= (*k); j++ {
			for i = 1; i <= (*m); i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		goblas.Zgemm(NoTrans, NoTrans, m, toPtr((*n)-(*l)), k, toPtrc128(-one), work, ldwork, v, ldv, &one, b, ldb)
		goblas.Zgemm(NoTrans, NoTrans, m, l, toPtr((*k)-(*l)), toPtrc128(-one), work.Off(0, kp-1), ldwork, v.Off(kp-1, np-1), ldv, &one, b.Off(0, np-1), ldb)
		goblas.Ztrmm(Right, Lower, NoTrans, NonUnit, m, l, &one, v.Off(0, np-1), ldv, work, ldwork)
		for j = 1; j <= (*l); j++ {
			for i = 1; i <= (*m); i++ {
				b.Set(i-1, (*n)-(*l)+j-1, b.Get(i-1, (*n)-(*l)+j-1)-work.Get(i-1, j-1))
			}
		}

		// ---------------------------------------------------------------------------
	} else if row && backward && left {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ V I ] ( I is K-by-K, V is K-by-M )
		//
		//        Form  H C  or  H**H C  where  C = [ B ]  (M-by-N)
		//                                          [ A ]  (K-by-N)
		//
		//        H = I - W**H T W          or  H**H = I - W**H T**H W
		//
		//        A = A -     T (A + V B)  or  A = A -     T**H (A + V B)
		//        B = B - V**H T (A + V B)  or  B = B - V**H T**H (A + V B)
		//
		// ---------------------------------------------------------------------------
		mp = minint((*l)+1, *m)
		kp = minint((*k)-(*l)+1, *k)

		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*l); i++ {
				work.Set((*k)-(*l)+i-1, j-1, b.Get(i-1, j-1))
			}
		}
		goblas.Ztrmm(Left, Upper, NoTrans, NonUnit, l, n, &one, v.Off(kp-1, 0), ldv, work.Off(kp-1, 0), ldwork)
		goblas.Zgemm(NoTrans, NoTrans, l, n, toPtr((*m)-(*l)), &one, v.Off(kp-1, mp-1), ldv, b.Off(mp-1, 0), ldb, &one, work.Off(kp-1, 0), ldwork)
		goblas.Zgemm(NoTrans, NoTrans, toPtr((*k)-(*l)), n, m, &one, v, ldv, b, ldb, &zero, work, ldwork)

		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*k); i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		goblas.Ztrmm(Left, Lower, mat.TransByte(trans), NonUnit, k, n, &one, t, ldt, work, ldwork)

		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*k); i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		goblas.Zgemm(ConjTrans, NoTrans, toPtr((*m)-(*l)), n, k, toPtrc128(-one), v.Off(0, mp-1), ldv, work, ldwork, &one, b.Off(mp-1, 0), ldb)
		goblas.Zgemm(ConjTrans, NoTrans, l, n, toPtr((*k)-(*l)), toPtrc128(-one), v, ldv, work, ldwork, &one, b, ldb)
		goblas.Ztrmm(Left, Upper, ConjTrans, NonUnit, l, n, &one, v.Off(kp-1, 0), ldv, work.Off(kp-1, 0), ldwork)
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*l); i++ {
				b.Set(i-1, j-1, b.Get(i-1, j-1)-work.Get((*k)-(*l)+i-1, j-1))
			}
		}

		// ---------------------------------------------------------------------------
	} else if row && backward && right {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ V I ] ( I is K-by-K, V is K-by-N )
		//
		//        Form  C H  or  C H**H  where  C = [ B A ] (A is M-by-K, B is M-by-N)
		//
		//        H = I - W**H T W            or  H**H = I - W**H T**H W
		//
		//        A = A - (A + B V**H) T      or  A = A - (A + B V**H) T**H
		//        B = B - (A + B V**H) T V    or  B = B - (A + B V**H) T**H V
		//
		// ---------------------------------------------------------------------------
		np = minint((*l)+1, *n)
		kp = minint((*k)-(*l)+1, *k)

		for j = 1; j <= (*l); j++ {
			for i = 1; i <= (*m); i++ {
				work.Set(i-1, (*k)-(*l)+j-1, b.Get(i-1, j-1))
			}
		}
		goblas.Ztrmm(Right, Upper, ConjTrans, NonUnit, m, l, &one, v.Off(kp-1, 0), ldv, work.Off(0, kp-1), ldwork)
		goblas.Zgemm(NoTrans, ConjTrans, m, l, toPtr((*n)-(*l)), &one, b.Off(0, np-1), ldb, v.Off(kp-1, np-1), ldv, &one, work.Off(0, kp-1), ldwork)
		goblas.Zgemm(NoTrans, ConjTrans, m, toPtr((*k)-(*l)), n, &one, b, ldb, v, ldv, &zero, work, ldwork)

		for j = 1; j <= (*k); j++ {
			for i = 1; i <= (*m); i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		goblas.Ztrmm(Right, Lower, mat.TransByte(trans), NonUnit, m, k, &one, t, ldt, work, ldwork)

		for j = 1; j <= (*k); j++ {
			for i = 1; i <= (*m); i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		goblas.Zgemm(NoTrans, NoTrans, m, toPtr((*n)-(*l)), k, toPtrc128(-one), work, ldwork, v.Off(0, np-1), ldv, &one, b.Off(0, np-1), ldb)
		goblas.Zgemm(NoTrans, NoTrans, m, l, toPtr((*k)-(*l)), toPtrc128(-one), work, ldwork, v, ldv, &one, b, ldb)
		goblas.Ztrmm(Right, Upper, NoTrans, NonUnit, m, l, &one, v.Off(kp-1, 0), ldv, work.Off(0, kp-1), ldwork)
		for j = 1; j <= (*l); j++ {
			for i = 1; i <= (*m); i++ {
				b.Set(i-1, j-1, b.Get(i-1, j-1)-work.Get(i-1, (*k)-(*l)+j-1))
			}
		}

	}
}
