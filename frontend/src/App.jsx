import { useState } from 'react'
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query'
import ProspectForm from './components/ProspectForm'
import Universe from './components/Universe'
import './index.css'

const queryClient = new QueryClient()

function AppInner() {
  const [appState, setAppState] = useState('FORM') // 'FORM' | 'UNIVERSE'
  const [formValues, setFormValues] = useState(null)

  const playersQuery = useQuery({
    queryKey: ['players'],
    queryFn: async () => {
      const res = await fetch('http://localhost:8000/players')
      if (!res.ok) throw new Error('Failed to fetch players')
      return res.json()
    },
    staleTime: Infinity,
  })

  const prospects2026Query = useQuery({
    queryKey: ['prospects2026'],
    queryFn: async () => {
      const res = await fetch('http://localhost:8000/prospects/2026')
      if (!res.ok) throw new Error('Failed to fetch 2026 prospects')
      return res.json()
    },
    staleTime: Infinity,
  })

  const handleSubmit = (values) => {
    setFormValues(values)
    setAppState('UNIVERSE')
  }

  const playersReady = playersQuery.status === 'success'

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      {/* Universe sits behind form, initialized only when players loaded */}
      {playersReady && (
        <Universe
          players={playersQuery.data.players}
          visible={appState === 'UNIVERSE'}
          prospectData={formValues}
        />
      )}

      {/* Form layer */}
      <ProspectForm
        visible={appState === 'FORM'}
        playersReady={playersReady}
        medians={playersQuery.data?.medians ?? {}}
        prospects2026={prospects2026Query.data?.prospects ?? []}
        onSubmit={handleSubmit}
      />
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppInner />
    </QueryClientProvider>
  )
}
